import enum
import logging
import argparse
import os
from urllib.parse import urlparse
import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Generator, Iterable, Iterator, List, Optional, Tuple

import grpc

from generated.salesforce.hyperdb.grpc.v1 import hyper_service_pb2 as hs_pb2
from generated.salesforce.hyperdb.grpc.v1 import hyper_service_pb2_grpc as hs_grpc
from generated.salesforce.hyperdb.grpc.v1 import error_details_pb2 as ed_pb2


logger = logging.getLogger(__name__)
class HyperGrpcError(Exception):
    def __init__(
        self,
        *,
        code: grpc.StatusCode,
        details: Optional[str] = None,
        sqlstate: Optional[str] = None,
        primary_message: Optional[str] = None,
        customer_detail: Optional[str] = None,
        customer_hint: Optional[str] = None,
        error_source: Optional[str] = None,
    ) -> None:
        self.code = code
        self.details = details
        self.sqlstate = sqlstate
        self.primary_message = primary_message
        self.customer_detail = customer_detail
        self.customer_hint = customer_hint
        self.error_source = error_source
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [f"code={self.code.name}"]
        if self.sqlstate:
            parts.append(f"sqlstate={self.sqlstate}")
        if self.primary_message:
            parts.append(f"message={self.primary_message}")
        elif self.details:
            parts.append(f"details={self.details}")
        if self.customer_detail:
            parts.append(f"detail={self.customer_detail}")
        if self.customer_hint:
            parts.append(f"hint={self.customer_hint}")
        if self.error_source:
            parts.append(f"source={self.error_source}")
        return ", ".join(parts)


def _convert_grpc_error(exc: grpc.RpcError) -> HyperGrpcError:
    """Convert grpc.RpcError with structured details to HyperGrpcError."""
    code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.UNKNOWN
    details_text = exc.details() if hasattr(exc, "details") else None
    # Parse structured details from trailers (grpc-status-details-bin)
    sqlstate = None
    primary_message = None
    customer_detail = None
    customer_hint = None
    error_source = None
    try:
        # Lazy import to avoid hard dependency at module import time
        from google.rpc import status_pb2 as google_status_pb2  # type: ignore
        md = dict(exc.trailing_metadata() or [])  # type: ignore[arg-type]
        raw = md.get("grpc-status-details-bin")
        if isinstance(raw, str):
            raw = raw.encode("latin1", errors="ignore")
        if raw:
            st = google_status_pb2.Status()
            st.MergeFromString(raw)
            for any_msg in st.details:
                info = ed_pb2.ErrorInfo()
                if any_msg.Is(info.DESCRIPTOR) and any_msg.Unpack(info):
                    sqlstate = info.sqlstate or None
                    primary_message = info.primary_message or None
                    customer_detail = info.customer_detail or None
                    customer_hint = info.customer_hint or None
                    error_source = info.error_source or None
                    break
    except Exception:  # Fall back to plain details
        pass
    return HyperGrpcError(
        code=code,
        details=details_text,
        sqlstate=sqlstate,
        primary_message=primary_message,
        customer_detail=customer_detail,
        customer_hint=customer_hint,
        error_source=error_source,
    )



@dataclass
class ResultChunk:
    data: bytes
    row_count: int


class HyperGrpcClient:
    """
    Thin gRPC client for HyperService with convenience helpers.

    - Provides ExecuteQuery, GetQueryInfo, GetQueryResult, CancelQuery wrappers
    - Use AdaptiveQueryResultIterator to stream results in ADAPTIVE mode
    """

    def __init__(self, channel: grpc.Channel, default_metadata: Optional[List[Tuple[str, str]]] = None):
        self._channel = channel
        self._stub = hs_grpc.HyperServiceStub(channel)
        self._default_metadata = tuple(default_metadata or [])

    @classmethod
    def secure_channel(
        cls,
        target: str,
        credentials: Optional[grpc.ChannelCredentials] = None,
        default_metadata: Optional[List[Tuple[str, str]]] = None,
    ) -> "HyperGrpcClient":
        creds = credentials or grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(target, creds)
        return cls(channel, default_metadata)

    def execute_query(
        self,
        sql: str,
        *,
        metadata: Optional[List[Tuple[str, str]]] = None,
        grpc_timeout: Optional[dt.timedelta] = None,
    ) -> Iterator[hs_pb2.ExecuteQueryResponse]:
        params = hs_pb2.QueryParam(
            query=sql,
            output_format=hs_pb2.OutputFormat.ARROW_IPC,
            transfer_mode=hs_pb2.QueryParam.TransferMode.ADAPTIVE,
        )

        md = self._merge_metadata(metadata)
        logger.debug("ExecuteQuery")
        grpc_timeout = None if grpc_timeout is None else grpc_timeout.total_seconds()
        return self._stub.ExecuteQuery(params, metadata=md, timeout=grpc_timeout)

    def get_query_info(
        self,
        query_id: str,
        *,
        streaming: bool = True,
        metadata: Optional[List[Tuple[str, str]]] = None,
        grpc_timeout: Optional[dt.timedelta] = None,
    ) -> Iterator[hs_pb2.QueryInfo]:
        request = hs_pb2.QueryInfoParam(query_id=query_id, streaming=streaming)
        md = self._merge_metadata([("x-hyperdb-query-id", query_id)], metadata)
        grpc_timeout = None if grpc_timeout is None else grpc_timeout.total_seconds()
        return self._stub.GetQueryInfo(request, metadata=md, timeout=grpc_timeout)

    def get_query_result(
        self,
        query_id: str,
        *,
        chunk_id: Optional[int] = None,
        omit_schema: bool = True,
        metadata: Optional[List[Tuple[str, str]]] = None,
        grpc_timeout: Optional[dt.timedelta] = None,
    ) -> Iterator[hs_pb2.QueryResult]:
        request = hs_pb2.QueryResultParam(
            query_id=query_id,
            output_format=hs_pb2.OutputFormat.ARROW_IPC,
            omit_schema=omit_schema,
        )
        if chunk_id is not None:
            request.chunk_id = int(chunk_id)
        md = self._merge_metadata([("x-hyperdb-query-id", query_id)], metadata)
        grpc_timeout = None if grpc_timeout is None else grpc_timeout.total_seconds()
        return self._stub.GetQueryResult(request, metadata=md, timeout=grpc_timeout)

    def cancel_query(self, query_id: str, metadata: Optional[List[Tuple[str, str]]] = None) -> None:
        request = hs_pb2.CancelQueryParam(query_id=query_id)
        md = self._merge_metadata([("x-hyperdb-query-id", query_id)], metadata)
        # Fire-and-forget; ignore result
        try:
            self._stub.CancelQuery(request, metadata=md)
        except grpc.RpcError as e:
            # Cancellation may race with query completion; not fatal
            logger.debug("CancelQuery failed: %s", e)

    def adaptive_iterator(
        self,
        sql: str,
        *,
        metadata: Optional[List[Tuple[str, str]]] = None,
        grpc_timeout: Optional[dt.timedelta] = None,
    ) -> "AdaptiveQueryResultIterator":
        execute_stream = self.execute_query(
            sql,
            metadata=metadata,
            grpc_timeout=grpc_timeout,
        )
        return AdaptiveQueryResultIterator(client=self, execute_stream=execute_stream)

    def _merge_metadata(
        self,
        *metadata: Optional[Iterable[Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        md: List[Tuple[str, str]] = list(self._default_metadata)
        for m in metadata:
            if m:
                md.extend(list(m))
        return md


class _State(enum.Enum):
    PROCESS_EXECUTE_QUERY_STREAM = enum.auto()
    CHECK_FOR_MORE_DATA = enum.auto()
    PROCESS_QUERY_RESULT_STREAM = enum.auto()
    PROCESS_QUERY_INFO_STREAM = enum.auto()
    COMPLETED = enum.auto()


@dataclass
class _Context:
    query_id: Optional[str] = None
    status: Optional[hs_pb2.QueryStatus] = None
    high_water: int = 1  # next chunk id to request

    # active streams
    execute_stream: Optional[Iterator[hs_pb2.ExecuteQueryResponse]] = None
    info_stream: Optional[Iterator[hs_pb2.QueryInfo]] = None
    result_stream: Optional[Iterator[hs_pb2.QueryResult]] = None

    # buffered results to yield next
    result_queue: Deque[hs_pb2.QueryResult] = field(default_factory=deque)

    def has_more_chunks(self) -> bool:
        return bool(self.status and (self.high_water < int(self.status.chunk_count)))

    def all_results_produced(self) -> bool:
        if not self.status:
            return False
        return self.status.completion_status in (
            hs_pb2.QueryStatus.CompletionStatus.RESULTS_PRODUCED,
            hs_pb2.QueryStatus.CompletionStatus.FINISHED,
        )


class AdaptiveQueryResultIterator:
    """
    Implements the adaptive state machine inspired by the Java/C++ clients.

    Iterate over this object to receive `QueryResult` messages (ARROW_IPC or JSON_ARRAY parts).
    """

    def __init__(
        self,
        *,
        client: HyperGrpcClient,
        execute_stream: Iterator[hs_pb2.ExecuteQueryResponse],
    ) -> None:
        self._client = client
        self._state: _State = _State.PROCESS_EXECUTE_QUERY_STREAM
        self._ctx = _Context(execute_stream=execute_stream)

    @property
    def query_id(self) -> Optional[str]:
        return self._ctx.query_id

    @property
    def query_status(self) -> Optional[hs_pb2.QueryStatus]:
        return self._ctx.status

    def __iter__(self) -> Iterator[hs_pb2.QueryResult]:
        while True:
            # If we already have results buffered, yield them immediately
            if self._ctx.result_queue:
                yield self._ctx.result_queue.popleft()
                continue

            if self._state == _State.PROCESS_EXECUTE_QUERY_STREAM:
                try:
                    assert self._ctx.execute_stream is not None
                    response = next(self._ctx.execute_stream)
                    if response.HasField("query_info"):
                        self._update_query_context(response.query_info)
                    elif response.HasField("query_result"):
                        self._ctx.result_queue.append(response.query_result)
                    else:
                        if not response.optional:
                            raise RuntimeError(
                                "Received unexpected non-optional ExecuteQueryResponse"
                            )
                except StopIteration:
                    self._transition(_State.CHECK_FOR_MORE_DATA)
                except grpc.RpcError as exc:
                    if exc.code() == grpc.StatusCode.CANCELLED:
                        logger.warning(
                            "ExecuteQuery stream cancelled; retrying via status")
                        self._ctx.execute_stream = None
                        self._transition(_State.CHECK_FOR_MORE_DATA)
                    else:
                        raise _convert_grpc_error(exc)

            elif self._state == _State.CHECK_FOR_MORE_DATA:
                if self._ctx.has_more_chunks():
                    chunk_id = self._ctx.high_water
                    self._ctx.high_water += 1
                    assert self._ctx.query_id
                    self._ctx.result_stream = self._client.get_query_result(
                        self._ctx.query_id,
                        chunk_id=chunk_id,
                        omit_schema=True,
                    )
                    self._transition(_State.PROCESS_QUERY_RESULT_STREAM)
                elif not self._ctx.all_results_produced():
                    assert self._ctx.query_id
                    self._ctx.info_stream = self._client.get_query_info(
                        self._ctx.query_id, streaming=True)
                    self._transition(_State.PROCESS_QUERY_INFO_STREAM)
                else:
                    self._transition(_State.COMPLETED)

            elif self._state == _State.PROCESS_QUERY_RESULT_STREAM:
                try:
                    assert self._ctx.result_stream is not None
                    result = next(self._ctx.result_stream)
                    self._ctx.result_queue.append(result)
                except StopIteration:
                    self._transition(_State.CHECK_FOR_MORE_DATA)
                except grpc.RpcError as exc:
                    if exc.code() == grpc.StatusCode.CANCELLED:
                        logger.warning(
                            "GetQueryResult stream cancelled; retrying")
                        self._ctx.result_stream = None
                        # Reset any partial results and retry the same chunk via CHECK_FOR_MORE_DATA
                        self._transition(_State.CHECK_FOR_MORE_DATA)
                    elif exc.code() == grpc.StatusCode.FAILED_PRECONDITION:
                        # Rely on GetQueryInfo to surface actual query error
                        self._drain_info_stream_if_open()
                        self._transition(_State.PROCESS_QUERY_INFO_STREAM)
                    else:
                        raise _convert_grpc_error(exc)

            elif self._state == _State.PROCESS_QUERY_INFO_STREAM:
                try:
                    progressed = False
                    while not self._ctx.has_more_chunks():
                        if self._ctx.info_stream is None:
                            break
                        try:
                            info = next(self._ctx.info_stream)
                            self._update_query_context(info)
                            progressed = True
                        except StopIteration:
                            self._ctx.info_stream = None
                            break
                    # Either chunks became available, or stream ended; re-check
                    self._transition(_State.CHECK_FOR_MORE_DATA)
                    # If we made progress by reading infos, loop continues
                except grpc.RpcError as exc:
                    if exc.code() == grpc.StatusCode.CANCELLED:
                        logger.warning(
                            "GetQueryInfo stream cancelled; retrying")
                        self._ctx.info_stream = None
                        self._transition(_State.CHECK_FOR_MORE_DATA)
                    else:
                        raise _convert_grpc_error(exc)

            elif self._state == _State.COMPLETED:
                return

    def stream_arrow_ipc(self) -> Iterator[bytes]:
        for qr in self:
            if qr.HasField("binary_part"):
                yield qr.binary_part.data
            elif qr.HasField("string_part"):
                # For JSON_ARRAY format, users likely prefer textual chunks
                yield qr.string_part.data.encode("utf-8")

    def _update_query_context(self, info: hs_pb2.QueryInfo) -> None:
        if info.optional:
            return
        if info.HasField("query_status"):
            self._ctx.status = info.query_status
            if not self._ctx.query_id:
                self._ctx.query_id = info.query_status.query_id

    def _transition(self, new_state: _State) -> None:
        logger.debug("state transition: %s -> %s (qid=%s)",
                     self._state, new_state, self._ctx.query_id)
        self._state = new_state

    def _drain_info_stream_if_open(self) -> None:
        if self._ctx.info_stream is None:
            return
        try:
            for _ in self._ctx.info_stream:
                pass
        except grpc.RpcError:
            # Ignore during drain
            pass
        finally:
            self._ctx.info_stream = None


__all__ = [
    "HyperGrpcClient",
    "AdaptiveQueryResultIterator",
    "ArrowIpcRowIterator",
    "OutputFormat",
    "ResultChunk",
]

# Public alias so callers can reference OutputFormat directly from this module
OutputFormat = hs_pb2.OutputFormat


class ArrowIpcRowIterator:
    """
    Helper that consumes an AdaptiveQueryResultIterator's Arrow IPC bytes and
    yields Python dict rows. It buffers the full Arrow IPC stream in memory.
    """

    def __init__(self, adaptive_iter: AdaptiveQueryResultIterator) -> None:
        self._adaptive_iter = adaptive_iter

    def iter_record_batches(self) -> Iterator[object]:
        """
        Stream-decode Arrow IPC without buffering all chunks in memory.

        This wraps the chunk iterator in a file-like object that feeds bytes
        on-demand to PyArrow's streaming reader.
        """
        try:
            import io
            import pyarrow as pa
            import pyarrow.ipc as pa_ipc
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "pyarrow is required for Arrow decoding. Install with `pip install pyarrow`."
            ) from e

        class _ChunkInput(io.RawIOBase):
            def __init__(self, chunk_iter: Iterator[bytes]) -> None:
                self._iter = iter(chunk_iter)
                self._buffer = bytearray()
                self._eof = False

            def readable(self) -> bool:
                return True

            def _fill_buffer(self) -> None:
                if self._eof:
                    return
                try:
                    next_chunk = next(self._iter)
                    if next_chunk:
                        self._buffer.extend(next_chunk)
                    else:
                        # Treat empty chunk as no-op; fetch next on subsequent reads
                        pass
                except StopIteration:
                    self._eof = True

            def readinto(self, b: bytearray) -> int:
                # Ensure at least some data is available or we reached EOF
                while not self._buffer and not self._eof:
                    self._fill_buffer()
                if not self._buffer and self._eof:
                    return 0
                n = min(len(b), len(self._buffer))
                b[:n] = self._buffer[:n]
                del self._buffer[:n]
                return n

            def read(self, size: int = -1) -> bytes:
                if size is None or size < 0:
                    chunks: list[bytes] = []
                    while True:
                        if self._buffer:
                            chunks.append(bytes(self._buffer))
                            self._buffer.clear()
                        if self._eof:
                            break
                        self._fill_buffer()
                        if not self._buffer and self._eof:
                            break
                    return b"".join(chunks)
                # Sized read
                out = bytearray()
                while len(out) < size:
                    if self._buffer:
                        take = min(size - len(out), len(self._buffer))
                        out += self._buffer[:take]
                        del self._buffer[:take]
                        continue
                    if self._eof:
                        break
                    self._fill_buffer()
                    if not self._buffer and self._eof:
                        break
                return bytes(out)

        stream = pa.input_stream(_ChunkInput(self._adaptive_iter.stream_arrow_ipc()))
        reader = pa_ipc.open_stream(stream)
        try:
            while True:
                batch = reader.read_next_batch()
                if batch is None:
                    break
                yield batch
        except StopIteration:
            return

    def iter_rows(self) -> Iterator[dict]:
        for batch in self.iter_record_batches():
            num_cols = batch.num_columns
            if num_cols == 0:
                continue
            # Preserve duplicate column names by disambiguating with numeric suffixes
            original_names = [batch.schema.names[i] for i in range(num_cols)]
            name_counts: dict[str, int] = {}
            output_names: list[str] = []
            for name in original_names:
                count = name_counts.get(name, 0) + 1
                name_counts[name] = count
                if count == 1:
                    output_names.append(name)
                else:
                    output_names.append(f"{name}_{count}")

            column_values_lists = [batch.column(i).to_pylist() for i in range(num_cols)]
            num_rows = len(column_values_lists[0]) if column_values_lists else 0
            for row_index in range(num_rows):
                yield {output_names[i]: column_values_lists[i][row_index] for i in range(num_cols)}

if __name__ == "__main__":
    # Lazy import to avoid hard dependency for library users
    from oauth import OAuthConfig, OAuthSession

    parser = argparse.ArgumentParser(description="Execute a SQL query via Hyper gRPC Adaptive iterator using OAuth Data Cloud session")
    parser.add_argument("--sql", required=False, default=os.getenv("HYPER_SQL", "SELECT 1"), help="SQL to execute")
    # output format fixed to ARROW_IPC
    # row/byte limits removed for simplicity
    parser.add_argument("--timeout", type=float, default=None, help="RPC timeout in seconds for calls")
    parser.add_argument("--print-rows", action="store_true", help="Decode Arrow IPC and print rows as Python dicts")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap when printing decoded rows (0 = no limit)")
    parser.add_argument("--metadata", action="append", default=[], help="Additional metadata headers key=value (repeatable)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    # Build default metadata from --metadata key=value
    md: list[tuple[str, str]] = []
    for item in args.metadata:
        if "=" in item:
            k, v = item.split("=", 1)
            md.append((k.strip(), v.strip()))
        elif item:
            logger.warning("Ignoring malformed metadata entry (expected key=value): %s", item)

    # Initialize OAuth Data Cloud session and derive secure gRPC target from instance URL
    cfg = OAuthConfig.from_env()
    base_session = OAuthSession(cfg)
    dc_session = base_session.create_dc_session()
    token = dc_session.get_token()
    instance_url = dc_session.get_instance_url()
    parsed = urlparse(instance_url if "://" in instance_url else f"https://{instance_url}")
    host = parsed.netloc or parsed.path
    if not host:
        raise RuntimeError(f"Invalid instance URL: {instance_url}")
    target = f"{host}:443"

    # Always secure; add Authorization metadata
    md.append(("authorization", f"Bearer {token}"))
    client = HyperGrpcClient.secure_channel(target, default_metadata=md)

    td_timeout = None if args.timeout is None else dt.timedelta(seconds=float(args.timeout))
    iterator = client.adaptive_iterator(
        args.sql,
        grpc_timeout=td_timeout,
    )

    total_parts = 0
    total_bytes = 0
    last_qid = None

    try:
        if args.print_rows:
            printed = 0
            row_iter = ArrowIpcRowIterator(iterator)
            for row in row_iter.iter_rows():
                if iterator.query_id and iterator.query_id != last_qid:
                    print(f"query_id={iterator.query_id}")
                    last_qid = iterator.query_id
                print(row)
                printed += 1
                if args.max_rows and printed >= args.max_rows:
                    break
        else:
            for part in iterator:
                if iterator.query_id and iterator.query_id != last_qid:
                    print(f"query_id={iterator.query_id}")
                    last_qid = iterator.query_id
                if part.HasField("binary_part"):
                    data = part.binary_part.data
                elif part.HasField("string_part"):
                    data = part.string_part.data.encode("utf-8")
                else:
                    data = b""
                total_parts += 1
                total_bytes += len(data)
                print(f"chunk {total_parts}: {len(data)} bytes, rows={part.result_part_row_count}")
    except grpc.RpcError as e:
        err = _convert_grpc_error(e)
        print(f"gRPC error: {err}")
        raise err

    print(f"done: parts={total_parts}, bytes={total_bytes}")
