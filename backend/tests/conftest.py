"""Test isolation: runs before any backend module is imported."""
import os
import tempfile

# Never write test runs / priority overrides into the real results database.
os.environ["HARP_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="harp_test_"), "test.db")
# The API tests make many compute calls from one client IP; the real limit is 10/min.
os.environ.setdefault("RATE_LIMIT_COMPUTE", "100000")
os.environ.setdefault("RATE_LIMIT_GENERAL", "100000")
