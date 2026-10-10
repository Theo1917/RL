"""TPC-H schema and data generator for query optimization benchmarks.

This module generates simplified TPC-H tables at configurable scale factors
for testing learned query optimizers against real PostgreSQL execution.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


@dataclass
class TPCHSchema:
    """TPC-H schema definition with foreign key relationships."""
    
    TABLES = {
        'nation': ['n_nationkey', 'n_name', 'n_regionkey'],
        'region': ['r_regionkey', 'r_name'],
        'customer': ['c_custkey', 'c_name', 'c_nationkey', 'c_acctbal'],
        'orders': ['o_orderkey', 'o_custkey', 'o_orderdate', 'o_totalprice'],
        'lineitem': ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_quantity', 'l_extendedprice'],
        'part': ['p_partkey', 'p_name', 'p_type', 'p_size'],
        'supplier': ['s_suppkey', 's_name', 's_nationkey', 's_acctbal'],
        'partsupp': ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost'],
    }
    
    FOREIGN_KEYS = [
        ('customer', 'c_nationkey', 'nation', 'n_nationkey'),
        ('orders', 'o_custkey', 'customer', 'c_custkey'),
        ('lineitem', 'l_orderkey', 'orders', 'o_orderkey'),
        ('lineitem', 'l_partkey', 'part', 'p_partkey'),
        ('lineitem', 'l_suppkey', 'supplier', 's_suppkey'),
        ('supplier', 's_nationkey', 'nation', 'n_nationkey'),
        ('partsupp', 'ps_partkey', 'part', 'p_partkey'),
        ('partsupp', 'ps_suppkey', 'supplier', 's_suppkey'),
        ('nation', 'n_regionkey', 'region', 'r_regionkey'),
    ]
    
    PRIMARY_KEYS = {
        'nation': 'n_nationkey',
        'region': 'r_regionkey',
        'customer': 'c_custkey',
        'orders': 'o_orderkey',
        'lineitem': ('l_orderkey', 'l_partkey', 'l_suppkey'),  # Composite
        'part': 'p_partkey',
        'supplier': 's_suppkey',
        'partsupp': ('ps_partkey', 'ps_suppkey'),  # Composite
    }


class TPCHGenerator:
    """Generate TPC-H data at specified scale factor."""
    
    # TPC-H standard scale factors
    BASE_COUNTS = {
        'region': 5,
        'nation': 25,
        'customer': 150000,
        'orders': 1500000,
        'lineitem': 6000000,
        'part': 200000,
        'supplier': 10000,
        'partsupp': 800000,
    }
    
    def __init__(self, scale_factor: float = 0.01, seed: int = 42):
        """Initialize generator.
        
        Args:
            scale_factor: TPC-H SF (0.01 = 10MB, 1 = 1GB, 10 = 10GB)
            seed: Random seed for reproducibility
        """
        self.scale_factor = scale_factor
        self.seed = seed
        self.rng = random.Random(seed)
        
    def row_count(self, table: str) -> int:
        """Calculate row count for table at current scale factor."""
        base = self.BASE_COUNTS[table]
        # Keep minimum counts for small scale factors
        return max(int(base * self.scale_factor), 10 if base > 100 else base)
    
    def generate_sql_schema(self) -> str:
        """Generate PostgreSQL DDL for TPC-H schema."""
        ddl = []
        
        # Create tables
        for table, columns in TPCHSchema.TABLES.items():
            col_defs = []
            pk = TPCHSchema.PRIMARY_KEYS[table]
            
            for col in columns:
                if 'key' in col:
                    col_type = 'INTEGER'
                elif 'date' in col:
                    col_type = 'DATE'
                elif 'bal' in col or 'price' in col or 'cost' in col:
                    col_type = 'DECIMAL(15,2)'
                elif 'qty' in col or 'size' in col:
                    col_type = 'INTEGER'
                else:
                    col_type = 'VARCHAR(255)'
                
                col_defs.append(f"    {col} {col_type}")
            
            # Add primary key constraint
            if isinstance(pk, tuple):
                pk_cols = ', '.join(pk)
                col_defs.append(f"    PRIMARY KEY ({pk_cols})")
            else:
                col_defs.append(f"    PRIMARY KEY ({pk})")
            
            table_sql = f"CREATE TABLE {table} (\n" + ',\n'.join(col_defs) + "\n);\n"
            ddl.append(table_sql)
        
        # Add foreign keys
        for child_table, child_col, parent_table, parent_col in TPCHSchema.FOREIGN_KEYS:
            fk_name = f"fk_{child_table}_{child_col}"
            fk_sql = (f"ALTER TABLE {child_table} "
                     f"ADD CONSTRAINT {fk_name} "
                     f"FOREIGN KEY ({child_col}) "
                     f"REFERENCES {parent_table}({parent_col});\n")
            ddl.append(fk_sql)
        
        # Add indexes on foreign keys for better join performance
        for child_table, child_col, _, _ in TPCHSchema.FOREIGN_KEYS:
            idx_sql = f"CREATE INDEX idx_{child_table}_{child_col} ON {child_table}({child_col});\n"
            ddl.append(idx_sql)
        
        return '\n'.join(ddl)
    
    def generate_insert_statements(self, table: str, batch_size: int = 1000) -> List[str]:
        """Generate INSERT statements for a table.
        
        Args:
            table: Table name
            batch_size: Rows per INSERT statement
            
        Returns:
            List of SQL INSERT statements
        """
        count = self.row_count(table)
        columns = TPCHSchema.TABLES[table]
        
        statements = []
        rows_generated = 0
        
        while rows_generated < count:
            batch = min(batch_size, count - rows_generated)
            values = []
            
            for i in range(batch):
                row_values = self._generate_row(table, rows_generated + i)
                values.append(f"({', '.join(row_values)})")
            
            insert_sql = (f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n" +
                         ',\n'.join(values) + ";\n")
            statements.append(insert_sql)
            rows_generated += batch
        
        return statements
    
    def _generate_row(self, table: str, row_id: int) -> List[str]:
        """Generate single row of data for table."""
        columns = TPCHSchema.TABLES[table]
        values = []
        
        for col in columns:
            if col.endswith('key'):
                # Primary or foreign key
                if col.startswith(table[0] + '_'):  # Primary key
                    values.append(str(row_id))
                elif col == 'c_nationkey':
                    values.append(str(self.rng.randint(0, self.row_count('nation') - 1)))
                elif col == 's_nationkey':
                    values.append(str(self.rng.randint(0, self.row_count('nation') - 1)))
                elif col == 'n_regionkey':
                    values.append(str(self.rng.randint(0, self.row_count('region') - 1)))
                elif col == 'o_custkey':
                    values.append(str(self.rng.randint(0, self.row_count('customer') - 1)))
                elif col == 'l_orderkey':
                    values.append(str(self.rng.randint(0, self.row_count('orders') - 1)))
                elif col == 'l_partkey' or col == 'ps_partkey':
                    values.append(str(self.rng.randint(0, self.row_count('part') - 1)))
                elif col == 'l_suppkey' or col == 'ps_suppkey':
                    values.append(str(self.rng.randint(0, self.row_count('supplier') - 1)))
                else:
                    values.append(str(row_id))
            
            elif 'date' in col:
                base_date = datetime(1992, 1, 1)
                days_offset = self.rng.randint(0, 2557)  # ~7 years
                date_val = base_date + timedelta(days=days_offset)
                values.append(f"'{date_val.strftime('%Y-%m-%d')}'")
            
            elif 'bal' in col or 'price' in col or 'cost' in col:
                amount = round(self.rng.uniform(100, 100000), 2)
                values.append(str(amount))
            
            elif 'qty' in col or 'size' in col:
                values.append(str(self.rng.randint(1, 100)))
            
            elif 'name' in col:
                names = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
                values.append(f"'{names[self.rng.randint(0, 4)]}_{row_id}'")
            
            elif 'type' in col:
                types = ['STANDARD', 'SMALL', 'MEDIUM', 'LARGE', 'ECONOMY']
                values.append(f"'{types[self.rng.randint(0, 4)]}'")
            
            else:
                values.append(f"'value_{row_id}'")
        
        return values
    
    def generate_full_script(self, include_analyze: bool = True) -> str:
        """Generate complete SQL script to create and populate TPC-H database.
        
        Args:
            include_analyze: Whether to add ANALYZE statements after data load
            
        Returns:
            Complete SQL script as string
        """
        script = [
            "-- TPC-H Schema and Data Generation",
            f"-- Scale Factor: {self.scale_factor}",
            f"-- Seed: {self.seed}",
            f"-- Generated: {datetime.now().isoformat()}",
            "\n-- Drop existing tables",
        ]
        
        # Drop in reverse FK order
        for table in reversed(list(TPCHSchema.TABLES.keys())):
            script.append(f"DROP TABLE IF EXISTS {table} CASCADE;")
        
        script.append("\n-- Create schema")
        script.append(self.generate_sql_schema())
        
        script.append("\n-- Load data")
        for table in TPCHSchema.TABLES.keys():
            script.append(f"\n-- Loading {table} ({self.row_count(table)} rows)")
            for insert in self.generate_insert_statements(table):
                script.append(insert)
        
        if include_analyze:
            script.append("\n-- Update statistics")
            for table in TPCHSchema.TABLES.keys():
                script.append(f"ANALYZE {table};")
        
        return '\n'.join(script)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get row counts for all tables at current scale factor."""
        return {table: self.row_count(table) for table in TPCHSchema.TABLES.keys()}


def main():
    """Demo: Generate TPC-H schema at small scale."""
    generator = TPCHGenerator(scale_factor=0.01, seed=42)
    
    print("TPC-H Table Row Counts (SF=0.01):")
    print("-" * 40)
    for table, count in generator.get_statistics().items():
        print(f"{table:15s}: {count:>8,} rows")
    
    print("\n\nGenerating SQL script...")
    script = generator.generate_full_script()
    
    output_file = "benchmarks/tpch_init.sql"
    with open(output_file, 'w') as f:
        f.write(script)
    
    print(f"✓ Generated {output_file} ({len(script)} bytes)")
    print("\nTo load into PostgreSQL:")
    print("  docker compose up -d")
    print(f"  psql -U postgres -d rl_optimizer -f {output_file}")


if __name__ == "__main__":
    main()
