-- The newest rows in the tables the API writes, refreshed every second.
-- From reference_app, in a terminal of its own:
--
--   docker compose exec -T postgres psql -U docchat < scripts/watch.sql
--
-- Ctrl+C stops it. Postgres logs no queries by default, so this is how you watch
-- it work: a turn adds two messages and maybe a cache row, an upload adds a
-- document that goes from pending to ready.
SELECT to_char(at, 'HH24:MI:SS') AS at, tbl, who, what, tokens
FROM (
    SELECT m.created_at AS at, 'messages' AS tbl, c.user_id AS who,
           m.role || ': ' || left(replace(m.content, E'\n', ' '), 60) AS what,
           m.n_in + m.n_out AS tokens
    FROM messages m JOIN conversations c ON c.id = m.conversation_id
    UNION ALL
    SELECT created_at, 'documents', status, title || ', ' || pages || ' pages', NULL
    FROM documents
    UNION ALL
    SELECT created_at, 'answer_cache', hits || ' hits', left(query, 60), n_tokens
    FROM answer_cache
) activity
ORDER BY activity.at DESC
LIMIT 12 \watch 1
