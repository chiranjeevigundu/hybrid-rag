# Orders API

Base URL: `https://api.example.com/v2`. All requests need a bearer token.

## Authentication

```http
POST /v2/auth/token
Content-Type: application/json

{"client_id": "...", "client_secret": "..."}
```

Tokens expire after 3600 seconds. Refresh before expiry; there is no grace period, and
a request with an expired token returns `401` with error code `token_expired` rather
than refreshing implicitly.

## Create an order

```http
POST /v2/orders
Authorization: Bearer <token>

{
  "sku": "WID-1024",
  "quantity": 3,
  "service": "EXP-2DA"
}
```

Returns `201` with the order body. The `idempotency_key` header is strongly
recommended — without it, a retried request after a network timeout creates a second
order, and there is no automatic deduplication.

## Retrieve an order

```http
GET /v2/orders/{order_id}
```

## Error codes

| Code | HTTP | Meaning                                    | Retry?          |
|------|------|--------------------------------------------|-----------------|
| 1001 | 400  | Malformed request body                     | No              |
| 1002 | 401  | Token expired or invalid                   | After refresh   |
| 1003 | 403  | Scope does not permit this operation       | No              |
| 1004 | 404  | Order not found                            | No              |
| 1005 | 409  | Idempotency key reused with different body | No              |
| 1006 | 429  | Rate limit exceeded                        | Yes, with backoff |
| 1007 | 503  | Upstream fulfilment system unavailable     | Yes, with backoff |

## Rate limits

600 requests per minute per client, measured in a sliding window. Exceeding it returns
`429` with a `Retry-After` header in seconds. The header is authoritative — clients
that ignore it and retry immediately are throttled more aggressively.

## Webhooks

Register an endpoint to receive `order.created`, `order.shipped`, and
`order.cancelled`. Deliveries are retried with exponential backoff for 24 hours.
Verify the `X-Signature` header before trusting a payload; it is an HMAC-SHA256 of the
raw request body using your webhook secret.
