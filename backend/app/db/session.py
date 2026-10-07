"""
Database session placeholder.

Reserved for database connection management when forensic job metadata,
evidence logs, and report persistence are introduced in subsequent steps.
"""


def get_db_session():
    """Dependency placeholder for database session injection."""
    yield None
