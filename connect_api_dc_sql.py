import json
import logging
from typing import Dict, List, Optional, Union

import anyio
import httpx

from auth_interface import AuthProvider

# Get logger for this module
logger = logging.getLogger(__name__)

# Default per-request HTTP timeout (seconds). Covers connect + read + write.
# Must be larger than the Data Cloud long-poll wait time (10s) plus headroom.
_DEFAULT_HTTP_TIMEOUT_S = 120.0


def _handle_error_response(response: httpx.Response):
    if response.status_code >= 300:
        # Parse error message from response
        message = response.text
        try:
            payload = json.loads(response.text)
            # Connect API error format: list with first element containing JSON
            # string in "message"
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
            response.reason_phrase,
            message,
        )


async def run_query(
    auth_provider: AuthProvider,
    sql: str,
    sql_parameters: Optional[List[Dict[str, Union[str, int, float, bool]]]] = None,
    dataspace: str = "default",
    workload_name: str | None = "data-360-mcp-query-oss",
    pagination_batch_size: int = 100000,
    query_settings: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[List, str]]:
    """
    Execute a SQL query using the Data Cloud Query Connect API, handling long-running queries
    and paginated result retrieval.

    This is a native coroutine that uses ``httpx.AsyncClient`` for non-blocking I/O, so it
    can be awaited from an asyncio/anyio event loop without tying up a worker thread while
    waiting on the server. ``AuthProvider`` methods are still synchronous; any potentially
    blocking work they perform (subprocesses, OAuth flows) is offloaded to a worker thread
    via ``anyio.to_thread.run_sync`` so it does not block the event loop either.

    Args:
        auth_provider: Authentication provider (SF CLI, OAuth, etc.)
        sql: SQL query string (use :paramName placeholders for parameterized queries)
        sql_parameters: Optional list of parameter dicts, each with "name" and "value" keys,
            and an optional "type" key. Values can be str, int, float, or bool. Examples:
            [{"name": "startDate", "value": "2025-01-01T00:00:00Z"}]
            [{"name": "limit", "value": 100}]
            [{"name": "active", "value": True}]
        dataspace: Data Cloud dataspace name (default: "default")
        workload_name: Workload name for resource management
        pagination_batch_size: Number of rows to fetch per page
        query_settings: Optional dict of query execution settings passed through as
            ``querySettings`` in the Data Cloud Query API request body. Example:
            {"query_timeout": "1800000ms"}

    Returns a dictionary containing:
    - 'data': the complete list of rows (aggregated across all pages) or "(empty)" if no rows
    - 'metadata': the schema/metadata of the result columns
    """
    base_url = await anyio.to_thread.run_sync(auth_provider.get_instance_url)
    token = await anyio.to_thread.run_sync(auth_provider.get_token)

    headers = {"Authorization": f"Bearer {token}"}
    url_base = base_url + "/services/data/v63.0/ssot/query-sql"
    common_params: dict[str, str] = {"dataspace": dataspace}
    if workload_name:
        common_params["workloadName"] = workload_name

    submit_body: dict = {"sql": sql}
    if sql_parameters:
        submit_body["sqlParameters"] = sql_parameters
    if query_settings:
        submit_body["querySettings"] = query_settings

    async with httpx.AsyncClient(timeout=_DEFAULT_HTTP_TIMEOUT_S) as client:
        # Step 1: submit the query
        logger.info(
            f"Submitting SQL query to {url_base}, with params: {common_params}")

        submit_response = await client.post(
            url_base,
            json=submit_body,
            params=common_params,
            headers=headers,
        )

        logger.info(
            f"Query submission response: status={submit_response.status_code}, "
            f"elapsed={submit_response.elapsed.total_seconds():.2f}s")
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
            # Signal that we want to do long-polling to get best latency for query
            # end notification and minimize RPC calls
            poll_params.update({
                "waitTimeMs": 10000,
            })
            poll_response = await client.get(
                poll_url, params=poll_params, headers=headers)

            logger.debug(
                f"Poll response: status={poll_response.status_code}, "
                f"elapsed={poll_response.elapsed.total_seconds():.2f}s")
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

            rows_response = await client.get(
                rows_url, params=rows_params, headers=headers)

            logger.debug(
                f"Rows fetch response: status={rows_response.status_code}, "
                f"elapsed={rows_response.elapsed.total_seconds():.2f}s")
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
    # Manual smoke test: authenticates and runs a query that spans multiple
    # pagination batches. Expects SF_ORG_ALIAS or SF_CLIENT_ID/SF_CLIENT_SECRET
    # to be set in the environment.
    import asyncio

    from auth_factory import create_auth_provider

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    async def _main():
        auth_provider = create_auth_provider()
        result = await run_query(
            auth_provider,
            "SELECT g::text || rpad(1::text,100) as a, g as b FROM generate_series(1, 40000) g ORDER BY b DESC",
        )
        print(f"Rows: {len(result['data'])}, Columns: {len(result['metadata'])}")

    asyncio.run(_main())
