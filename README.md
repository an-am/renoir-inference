# Distributed Real-Time Inference pipeline using Renoir in Rust

Renoir is a **distributed data processing platform**, 
based on the **dataflow paradigm**, that provides an ergonomic programming interface, similar to that of Apache Flink, but has **much better performance** characteristics. More infos on Renoir [here].(https://databrush.it/renoir/overview/)

Input is sourced from a **PostgreSQL database**, which holds personal and financial data
of users (from now, clients). The system must listen for insertions in a DB table on a
dedicated channel, retrieving and processing client records as soon as they are inserted
into the DB; thus, each record insertion is considered as an inference request.

The design of this database-driven pipeline introduces **three practical concerns**, which
inform the framework comparison: (1) it enables evaluating how different frameworks
manage database connections from distributed workers; (2) with client’sdata, it ispossible
to implement feature enrichment on-the-fly (computing a new client feature based on
existing features); and (3) it provides a natural way to implement **per-client state**.

In fact, there are multiple record insertions in the DB by the same client, and the state
tracks (that is, ***counts***) the number of inference requests received per client. This also
acts as a filtering mechanism: once a client exceeds a fixed threshold, their requests are
no longer processed. This stateful component also provides an opportunity to evaluate
how each framework supports stateful logic, both in terms of programming complexity
and its performance implications.

Valid requests then proceed to a preprocessing stage, where raw features are transformed
in the tensors the model expects. The inference step uses a **pre-trained Keras neural network**: each worker keeps a local copy of the model in memory, exploiting **data parallelism**
to avoid repeated I/O and reduce inference latency.

Finally, the pipeline concludes with **two database accesses**. First, the client’s record is
updated with the predicted value and enriched feature. Second, the system queries a separate financial product table, on the same DB, to retrieve personalized recommendations
based on the updated data.


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
