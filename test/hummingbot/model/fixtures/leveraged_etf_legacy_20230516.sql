PRAGMA journal_mode = DELETE;

CREATE TABLE Metadata (
    key TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT INTO Metadata (key, value)
VALUES ('local_db_version', '20230516');

INSERT INTO Metadata (key, value)
VALUES ('legacy_sentinel', 'preserve-me');

CREATE TABLE Executors (
    id TEXT NOT NULL PRIMARY KEY,
    timestamp FLOAT NOT NULL,
    type TEXT NOT NULL,
    close_type INTEGER,
    close_timestamp BIGINT,
    status INTEGER NOT NULL,
    config JSON NOT NULL,
    net_pnl_pct FLOAT NOT NULL,
    net_pnl_quote FLOAT NOT NULL,
    cum_fees_quote FLOAT NOT NULL,
    filled_amount_quote FLOAT NOT NULL,
    is_active BOOLEAN NOT NULL,
    is_trading BOOLEAN NOT NULL,
    custom_info JSON NOT NULL,
    controller_id TEXT
);

INSERT INTO Executors (
    id,
    timestamp,
    type,
    close_type,
    close_timestamp,
    status,
    config,
    net_pnl_pct,
    net_pnl_quote,
    cum_fees_quote,
    filled_amount_quote,
    is_active,
    is_trading,
    custom_info,
    controller_id
) VALUES (
    'legacy-position-executor',
    1710000000.0,
    'position_executor',
    NULL,
    NULL,
    1,
    '{"id":"legacy-position-executor","type":"position_executor"}',
    0.0,
    0.0,
    0.0,
    0.0,
    1,
    0,
    '{"legacy":true}',
    'legacy-controller'
);

CREATE INDEX ex_type ON Executors (type);
CREATE INDEX ex_type_timestamp ON Executors (type, timestamp);
CREATE INDEX ex_timestamp ON Executors (timestamp);
CREATE INDEX ex_close_timestamp ON Executors (close_timestamp);
CREATE INDEX ex_status ON Executors (status);
CREATE INDEX ex_type_status ON Executors (type, status);
