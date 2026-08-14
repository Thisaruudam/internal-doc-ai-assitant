"""Generate a bcrypt hash for app/auth/users.yaml.

Usage:
    uv run python scripts/hash_password.py "new-password"
"""

import sys

from app.auth.users import hash_password

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: hash_password.py <password>")
    print(hash_password(sys.argv[1]))
