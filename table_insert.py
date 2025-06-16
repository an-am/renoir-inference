import psycopg
import random
from pandas import DataFrame
import sys

def start_table_insert(num_requests):
    db_config = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "localhostpostgrespass",
        "host": "localhost",
        "port": "5432"
    }

    # Establish connection
    conn = psycopg.connect("postgresql://postgres:localhostpostgrespass@localhost:5432/postgres")
    # conn.set_isolation_level(psycopg.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    data = []
    c_id = [i for i in range(1, 101)]
    for i in range(len(c_id)):
        data.append({
            "age": random.randint(18, 90),
            "gender": random.randint(0, 1),
            "family_members": random.randint(0, 5),
            "financial_education": random.uniform(0.036098897, 0.902932641),
            "income": random.uniform(1.537764666, 365.3233855),
            "wealth": random.uniform(1.057414979, 2233.228433),
            "income_investment": random.randint(0, 1),
            "accumulation_investment": random.randint(0, 1),
            "financial_status": 0,
            "client_id": c_id[i]
        })

    # Convert the list of dictionaries into a DataFrame
    df = DataFrame(data)

    for request in range(num_requests):
        client_id_index = request % 100  # Cyclic client ID index
        row = df.iloc[client_id_index]  # Select row for the cyclic client ID

        query = f"""INSERT INTO Needs (id, age, gender, family_members, financial_education,
                                        income, wealth, income_investment, accumulation_investment, financial_status, client_id)
                        VALUES 
                        (
                            (SELECT COALESCE(MAX(ID), 0) + 1 FROM Needs), 
                            {row['age']}, 
                            {row['gender']}, 
                            {row['family_members']},
                            {row['financial_education']},
                            {row['income']},
                            {row['wealth']},
                            {row['income_investment']},
                            {row['accumulation_investment']},
                            {row['financial_status']},
                            {row['client_id']}
                        )"""
        cursor.execute(query)
        print(".", end="")
    print()

    cursor.close()
    conn.commit()
    conn.close()

if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    start_table_insert(num)