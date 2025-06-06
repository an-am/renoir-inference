1) In Postgres, inserire questo trigger:
```
create trigger needs_insert_trigger
    after insert
    on needs
    for each row
execute procedure notify_needs_insert();
```
e questa funzione:
```
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
2) Eseguire Renoir.
3) Con ```start_table_insert(1000)``` di ```table_insert.py``` si inseriscono 1000 tuple nel db.
