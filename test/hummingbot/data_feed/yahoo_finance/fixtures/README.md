# Yahoo chart fixtures

These chart responses are hand-authored, sanitized synthetic fixtures shaped from
Yahoo's public `/v8/finance/chart/{symbol}` response contract. They contain no
cookies, crumb, credentials, customer data, or response captured from a private
account. Tests never contact Yahoo.

The adjusted-close and market-price values intentionally differ from the daily
regular `quote.close` so a fallback to either field is detectable.
