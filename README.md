1) In Postgres, creare la tabella needs:

```sql
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
```

e la tabella products:
```sql
create table products
(
    id_product   integer,
    income       integer,
    accumulation integer,
    risk         real,
    description  text
);
```

2) inserire questo trigger:
```sql
create trigger needs_insert_trigger
    after insert
    on needs
    for each row
execute procedure notify_needs_insert();
```
e questa funzione:
```sql
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
```
3) Eseguire Renoir.
4) Con ```start_table_insert(1000)``` di ```table_insert.py``` si inseriscono 1000 tuple nel db.
