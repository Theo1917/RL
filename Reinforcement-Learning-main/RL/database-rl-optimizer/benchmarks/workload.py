"""TPC-H query workload generator for benchmarking learned optimizers."""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class QueryTemplate:
    """TPC-H query template with join patterns."""
    name: str
    tables: List[str]
    join_predicates: List[Tuple[str, str, str, str]]  # (table1, col1, table2, col2)
    filter_predicates: List[str]
    description: str


class TPCHWorkload:
    """Generate TPC-H inspired join queries for benchmarking."""
    
    # Simplified TPC-H query patterns
    TEMPLATES = [
        QueryTemplate(
            name="Q3_CustomerOrders",
            tables=["customer", "orders", "lineitem"],
            join_predicates=[
                ("customer", "c_custkey", "orders", "o_custkey"),
                ("orders", "o_orderkey", "lineitem", "l_orderkey"),
            ],
            filter_predicates=[
                "o_orderdate < '1995-03-15'",
                "l_quantity > 10"
            ],
            description="Customer orders with line items (3 tables)"
        ),
        QueryTemplate(
            name="Q5_LocalSupplier",
            tables=["region", "nation", "customer", "orders", "lineitem", "supplier"],
            join_predicates=[
                ("region", "r_regionkey", "nation", "n_regionkey"),
                ("nation", "n_nationkey", "customer", "c_nationkey"),
                ("customer", "c_custkey", "orders", "o_custkey"),
                ("orders", "o_orderkey", "lineitem", "l_orderkey"),
                ("lineitem", "l_suppkey", "supplier", "s_suppkey"),
                ("supplier", "s_nationkey", "nation", "n_nationkey"),
            ],
            filter_predicates=[
                "r_name = 'ASIA'",
                "o_orderdate >= '1994-01-01'",
                "o_orderdate < '1995-01-01'"
            ],
            description="Regional supplier revenue (6 tables)"
        ),
        QueryTemplate(
            name="Q8_NationMarketShare",
            tables=["part", "supplier", "lineitem", "orders", "customer", "nation", "region"],
            join_predicates=[
                ("part", "p_partkey", "lineitem", "l_partkey"),
                ("supplier", "s_suppkey", "lineitem", "l_suppkey"),
                ("lineitem", "l_orderkey", "orders", "o_orderkey"),
                ("orders", "o_custkey", "customer", "c_custkey"),
                ("customer", "c_nationkey", "nation", "n_nationkey"),
                ("nation", "n_regionkey", "region", "r_regionkey"),
            ],
            filter_predicates=[
                "r_name = 'AMERICA'",
                "o_orderdate BETWEEN '1995-01-01' AND '1996-12-31'",
                "p_type = 'ECONOMY'"
            ],
            description="Nation market share (7 tables)"
        ),
        QueryTemplate(
            name="Q10_CustomerReturns",
            tables=["customer", "orders", "lineitem", "nation"],
            join_predicates=[
                ("customer", "c_custkey", "orders", "o_custkey"),
                ("orders", "o_orderkey", "lineitem", "l_orderkey"),
                ("customer", "c_nationkey", "nation", "n_nationkey"),
            ],
            filter_predicates=[
                "o_orderdate >= '1993-10-01'",
                "o_orderdate < '1994-01-01'",
                "l_quantity > 5"
            ],
            description="Customer returns analysis (4 tables)"
        ),
        QueryTemplate(
            name="Q11_SupplierPartValue",
            tables=["partsupp", "supplier", "nation"],
            join_predicates=[
                ("partsupp", "ps_suppkey", "supplier", "s_suppkey"),
                ("supplier", "s_nationkey", "nation", "n_nationkey"),
            ],
            filter_predicates=[
                "n_name = 'GERMANY'",
                "ps_supplycost > 100"
            ],
            description="Important stock identification (3 tables)"
        ),
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize workload generator.
        
        Args:
            seed: Random seed for query generation
        """
        self.seed = seed
        self.rng = random.Random(seed)
    
    def generate_query(self, template: QueryTemplate, parameterize: bool = True) -> str:
        """Generate SQL query from template.
        
        Args:
            template: Query template to instantiate
            parameterize: Whether to randomize filter predicates
            
        Returns:
            SQL query string
        """
        # Build SELECT clause
        select_cols = [f"{table}.{table[0]}_*" for table in template.tables[:2]]
        
        # Build FROM clause with explicit joins
        from_clause = template.tables[0]
        for table1, col1, table2, col2 in template.join_predicates:
            if table2 not in from_clause:
                from_clause += f"\n  JOIN {table2} ON {table1}.{col1} = {table2}.{col2}"
        
        # Build WHERE clause
        where_predicates = template.filter_predicates.copy()
        if parameterize:
            # Randomize some predicate values
            where_predicates = [self._randomize_predicate(pred) for pred in where_predicates]
        
        where_clause = "\n  AND ".join(where_predicates) if where_predicates else "1=1"
        
        # Assemble query
        query = f"""SELECT {', '.join(select_cols)}
FROM {from_clause}
WHERE {where_clause}"""
        
        return query
    
    def _randomize_predicate(self, predicate: str) -> str:
        """Randomize values in predicate for query diversity."""
        # Randomize dates
        if "1995-03-15" in predicate:
            year = self.rng.randint(1992, 1998)
            month = self.rng.randint(1, 12)
            day = self.rng.randint(1, 28)
            return predicate.replace("1995-03-15", f"{year:04d}-{month:02d}-{day:02d}")
        
        # Randomize quantities
        if "l_quantity >" in predicate:
            qty = self.rng.randint(5, 50)
            return f"l_quantity > {qty}"
        
        # Keep other predicates as-is
        return predicate
    
    def generate_workload(self, num_queries: int = 20, 
                         table_range: Tuple[int, int] = (3, 6)) -> List[Dict]:
        """Generate diverse workload of queries.
        
        Args:
            num_queries: Number of queries to generate
            table_range: (min_tables, max_tables) to include
            
        Returns:
            List of query dictionaries with metadata
        """
        workload = []
        min_tables, max_tables = table_range
        
        # Filter templates by table count
        valid_templates = [
            t for t in self.TEMPLATES 
            if min_tables <= len(t.tables) <= max_tables
        ]
        
        for i in range(num_queries):
            template = self.rng.choice(valid_templates)
            sql = self.generate_query(template, parameterize=True)
            
            workload.append({
                'query_id': i,
                'template': template.name,
                'num_tables': len(template.tables),
                'tables': template.tables,
                'sql': sql,
                'description': template.description
            })
        
        return workload
    
    def get_template(self, name: str) -> QueryTemplate:
        """Get template by name."""
        for template in self.TEMPLATES:
            if template.name == name:
                return template
        raise ValueError(f"Template '{name}' not found")
    
    def list_templates(self, table_count: int = None) -> List[str]:
        """List available templates, optionally filtered by table count."""
        templates = self.TEMPLATES
        if table_count:
            templates = [t for t in templates if len(t.tables) == table_count]
        return [t.name for t in templates]


def main():
    """Demo workload generation."""
    workload_gen = TPCHWorkload(seed=42)
    
    print("Available TPC-H Query Templates:")
    print("=" * 60)
    for template in TPCHWorkload.TEMPLATES:
        print(f"\n{template.name} ({len(template.tables)} tables)")
        print(f"  {template.description}")
        print(f"  Tables: {', '.join(template.tables)}")
    
    print("\n\nSample Generated Workload:")
    print("=" * 60)
    
    workload = workload_gen.generate_workload(num_queries=5, table_range=(3, 4))
    
    for query in workload:
        print(f"\nQuery {query['query_id']}: {query['template']}")
        print(f"Tables: {query['num_tables']}")
        print(f"SQL:\n{query['sql']}\n")


if __name__ == "__main__":
    main()
