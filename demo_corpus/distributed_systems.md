# Distributed Systems

A distributed system coordinates components that communicate across a network. Partial failure is fundamental because one component can fail while others continue operating.

## Reliability

Retries need bounded backoff and idempotency. A timeout prevents a caller from waiting forever, but it does not prove that the remote operation did not happen.
