create table needs
(
    id                      integer,
    age                     integer,
    gender                  integer,
    family_members          integer,
    financial_education     real,
    risk_propensity         real,
    income                  real,
    wealth                  real,
    income_investment       integer,
    accumulation_investment integer,
    financial_status        real,
    client_id               integer
);

create function notify_needs_insert() returns trigger
    language plpgsql
as
$$
DECLARE
    payload TEXT;
BEGIN
    -- Create the payload message
    payload := json_build_object('id', NEW.id, 'client_id', NEW.client_id)::TEXT;

    -- Send the notification
    PERFORM pg_notify('table_insert', payload);

    RETURN NEW;
END;
$$;

create trigger needs_insert_trigger
    after insert
    on needs
    for each row
execute procedure notify_needs_insert();