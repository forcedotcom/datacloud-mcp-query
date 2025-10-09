import json
import logging
import time
from typing import Dict, List, Optional, Union

import requests

from oauth import OAuthSession, OAuthConfig

# Get logger for this module
logger = logging.getLogger(__name__)


def _handle_error_response(response: requests.Response):
    if response.status_code >= 300:
        # Parse error message from response
        message = response.text
        try:
            payload = json.loads(response.text)
            # Connect API error format: list with first element containing JSON string in "message"
            if isinstance(payload, list) and len(payload) > 0:
                structured_message = payload[0]
                try:
                    errors_details_json = structured_message.get("message", "")
                    details = json.loads(
                        errors_details_json) if errors_details_json else None
                    if details:
                        message = errors_details_json
                except Exception:
                    pass
        except Exception:
            pass

        # Raise exception with error message
        raise Exception(
            response.status_code,
            response.reason,
            message,
        )


def run_query(
    oauth_session: OAuthSession,
    sql: str,
    dataspace: str = "default",
    workload_name: str | None = None,
    pagination_batch_size: int = 100000,
) -> Dict[str, Union[List, str]]:
    """
    Execute a SQL query using the Data Cloud Query Connect API, handling long-running queries
    and paginated result retrieval.

    Returns a dictionary containing:
    - 'data': the complete list of rows (aggregated across all pages) or "(empty)" if no rows
    - 'metadata': the schema/metadata of the result columns
    """
    base_url = oauth_session.get_instance_url()
    token = oauth_session.get_token()

    headers = {"Authorization": f"Bearer {token}"}
    url_base = base_url + "/services/data/v63.0/ssot/query-sql"
    common_params: dict[str, str] = {"dataspace": dataspace}
    if workload_name:
        common_params["workloadName"] = workload_name

    # Step 1: submit the query
    submit_body = {"sql": sql}
    logger.info(
        f"Submitting SQL query to {url_base}, with params: {common_params}")

    submit_response = requests.post(
        url_base, json=submit_body, params=common_params, headers=headers, timeout=100)

    logger.info(
        f"Query submission response: status={submit_response.status_code}, elapsed={submit_response.elapsed.total_seconds():.2f}s")
    _handle_error_response(submit_response)

    submit_payload = submit_response.json()
    status_obj = submit_payload.get("status", {})
    query_id = status_obj.get("queryId") or submit_payload.get("queryId")
    if not query_id:
        raise Exception(500, "MissingQueryId",
                        "Query ID not returned by the API.")

    # Collect initial rows and metadata if present
    rows: list = submit_payload.get("data", []) or []
    metadata = submit_payload.get("metadata", [])
    completion = status_obj.get("completionStatus")
    total_row_count = int(status_obj.get("rowCount"))

    # Step 2: poll for completion when needed (long-polling via waitTimeMs)
    poll_count = 0
    while completion not in ["Finished", "ResultsProduced"]:
        poll_count += 1
        poll_url = f"{url_base}/{query_id}"
        logger.debug(
            f"Polling query status (attempt {poll_count}): {poll_url}")

        poll_params = dict(common_params)
        # Signal that we want to do long-polling to get best latency for query end notification and minimize RPC calls
        poll_params.update({
            "waitTimeMs": 10000,
        })
        poll_response = requests.get(
            poll_url, params=poll_params, headers=headers, timeout=30)

        logger.debug(
            f"Poll response: status={poll_response.status_code}, elapsed={poll_response.elapsed.total_seconds():.2f}s")
        _handle_error_response(poll_response)
        poll_payload = poll_response.json()
        completion = poll_payload.get("completionStatus")
        total_row_count = int(poll_payload.get("rowCount"))

    # Step 3: retrieve remaining rows via pagination
    while len(rows) < total_row_count:
        rows_params = dict(common_params)
        rows_params.update({
            "rowLimit": pagination_batch_size,
            "offset": len(rows),
            "omitSchema": "true",
        })

        rows_url = f"{url_base}/{query_id}/rows"
        logger.debug(
            f"Fetching rows: offset={rows_params.get('offset')}, limit={rows_params.get('rowLimit')}")

        rows_response = requests.get(
            rows_url, params=rows_params, headers=headers, timeout=60)

        logger.debug(
            f"Rows fetch response: status={rows_response.status_code}, elapsed={rows_response.elapsed.total_seconds():.2f}s")
        _handle_error_response(rows_response)

        chunk = rows_response.json()
        chunk_rows = chunk.get("data", []) or []
        returned_rows = int(chunk.get("returnedRows", len(chunk_rows)))

        if returned_rows == 0:
            raise Exception(500, "MissingRows",
                            "Expected rows to be returned, but received 0.")

        rows.extend(chunk_rows)
        logger.debug(
            f"Retrieved {returned_rows} rows, total so far: {len(rows)}")

    logger.info(f"Query completed: retrieved {len(rows)} total rows")
    return {
        "data": rows,
        "metadata": metadata
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set debug level for this module during testing
    logger.setLevel(logging.DEBUG)

    sf_org: OAuthConfig = OAuthConfig.from_env()
    oauth_session: OAuthSession = OAuthSession(sf_org)

    result = run_query(OAuthSession(OAuthConfig.from_env(
    )), "SELECT g::text || rpad(1::text,100) as a, g as b FROM generate_series(1, 40000) g ORDER BY b DESC")
    print(f"Query result: {len(result['data'])} rows returned")
    print(result)
