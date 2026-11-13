# Automated Security Test File
DB_URL = "postgres://admin:SuperSecretPass123!@db.internal.corp:5432/production"

def query_user(username: str):
    # Intentional SQL injection string concatenation
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return query
