import os
import requests
import psycopg2
import sqlite3
from typing import Optional


class BaseCRMConnector:
    def __init__(self, mode: str = "api"):
        """
        Поддержка режимов: 'api', 'postgres', 'sqlite'
        """
        self.mode = mode.lower()

        if self.mode == "api":
            self.base_url = os.getenv("CRM_API_URL")
            self.token = os.getenv("CRM_API_TOKEN")

        elif self.mode == "postgres":
            self.db_connection = psycopg2.connect(
                dbname=os.getenv("CRM_DB_NAME"),
                user=os.getenv("CRM_DB_USER"),
                password=os.getenv("CRM_DB_PASSWORD"),
                host=os.getenv("CRM_DB_HOST"),
                port=os.getenv("CRM_DB_PORT")
            )

        elif self.mode == "sqlite":
            self.sqlite_path = os.getenv("CRM_SQLITE_PATH", "crm.db")
            self.db_connection = sqlite3.connect(self.sqlite_path)
            self.db_connection.row_factory = sqlite3.Row

        else:
            raise ValueError(f"Unsupported CRM mode: {self.mode}")

    def get_client_by_name(self, name: str) -> Optional[dict]:
        if self.mode == "api":
            response = requests.get(
                f"{self.base_url}/clients",
                params={"name": name},
                headers={"Authorization": f"Bearer {self.token}"}
            )
            if response.ok and response.json():
                return response.json()[0]

        elif self.mode in ("postgres", "sqlite"):
            with self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "SELECT * FROM clients WHERE full_name LIKE ? LIMIT 1",
                    (f"%{name}%",)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)

        return None

    def get_order_by_number(self, number: str) -> Optional[dict]:
        if self.mode == "api":
            response = requests.get(
                f"{self.base_url}/orders/{number}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            if response.ok:
                return response.json()

        elif self.mode in ("postgres", "sqlite"):
            with self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "SELECT * FROM orders WHERE order_number = ?",
                    (number,)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)

        return None

    def get_ticket_by_id(self, ticket_id: str) -> Optional[dict]:
        if self.mode == "api":
            response = requests.get(
                f"{self.base_url}/tickets/{ticket_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            if response.ok:
                return response.json()

        elif self.mode in ("postgres", "sqlite"):
            with self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "SELECT * FROM tickets WHERE id = ?",
                    (ticket_id,)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)

        return None
