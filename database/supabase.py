import os
from supabase import create_client, Client

class SupabaseManager:
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        supabase: Client = create_client(url, key)
        self.supabase = supabase

    def insert_data(self, table_name, data):
        """Insert data into a specified table"""
        response = self.supabase.table(table_name).insert(data).execute()
        return response
    
    # Update data in a specified table
    def update_data(self, table_name, data, insertedId, field=None):
        """Update data in a specified table"""
        # update only field
        if field:
            response = self.supabase.table(table_name).update({field: data}).eq("id", insertedId).execute()
        else:
            # update all fields
            response = self.supabase.table(table_name).update(data).eq("id", insertedId).execute()
            
        return response