"""
Synthetic database generation for RL query optimization.

This module generates synthetic relational schemas with:
- Tables, columns, primary keys, foreign keys
- Row counts and relationship information
- Join selectivity estimates

All generation is deterministic and reproducible using random seeds.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random


@dataclass
class Column:
    """Represents a column in a database table."""
    name: str
    data_type: str = "INTEGER"


@dataclass
class ForeignKey:
    """Represents a foreign key relationship."""
    column: str
    references_table: str
    references_column: str


@dataclass
class Table:
    """Represents a database table with schema and statistics."""
    name: str
    columns: List[Column]
    primary_key: str
    row_count: int
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    indexed_columns: List[str] = field(default_factory=list)

    def __hash__(self):
        """Make Table hashable for use in sets/dicts."""
        return hash(self.name)

    def __eq__(self, other):
        """Tables are equal if they have the same name."""
        if not isinstance(other, Table):
            return False
        return self.name == other.name


@dataclass
class Database:
    """
    Represents a synthetic relational database.

    Attributes:
        tables: List of tables in the database
        join_selectivities: Dict mapping table pairs to selectivity estimates
        seed: Random seed used for generation
    """
    tables: List[Table]
    join_selectivities: Dict[Tuple[str, str], float]
    seed: int

    def get_table(self, name: str) -> Table:
        """Get a table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        raise ValueError(f"Table '{name}' not found in database")

    def get_relationships(self) -> List[Tuple[str, str]]:
        """
        Get all foreign key relationships as table name pairs.
        Returns canonical ordered pairs (sorted alphabetically).
        """
        relationships = set()
        for table in self.tables:
            for fk in table.foreign_keys:
                # Canonical ordering: alphabetically sorted
                pair = tuple(sorted([table.name, fk.references_table]))
                relationships.add(pair)
        return sorted(list(relationships))

    def __str__(self) -> str:
        """Return a human-readable representation of the database."""
        lines = ["Synthetic Database", "-" * 50, ""]
        lines.append(f"Tables: {len(self.tables)}")
        lines.append(f"Seed: {self.seed}")
        lines.append("")

        for table in self.tables:
            lines.append(f"{table.name}")
            lines.append(f"  Rows: {table.row_count}")
            lines.append(f"  Primary Key: {table.primary_key}")

            for fk in table.foreign_keys:
                lines.append(f"  Foreign Key: {fk.column} -> {fk.references_table}.{fk.references_column}")

            lines.append("")

        if self.join_selectivities:
            lines.append("Join Selectivities")
            lines.append("-" * 50)
            for (t1, t2), sel in sorted(self.join_selectivities.items()):
                lines.append(f"{t1} <-> {t2}: {sel:.4f}")

        return "\n".join(lines)


def _generate_table_names(num_tables: int, rng: random.Random) -> List[str]:
    """
    Generate realistic table names for a relational schema.

    Uses common database entity names and ensures variety.
    """
    # Common database entity names
    base_names = [
        "Customer", "Order", "Product", "Employee", "Department",
        "Invoice", "Payment", "Shipment", "Supplier", "Category",
        "User", "Account", "Transaction", "Item", "Location",
        "Student", "Course", "Enrollment", "Instructor", "Assignment"
    ]

    # Shuffle and take the required number
    shuffled = base_names.copy()
    rng.shuffle(shuffled)

    return shuffled[:num_tables]


def _generate_connected_schema(num_tables: int, rng: random.Random) -> List[Tuple[int, int]]:
    """
    Generate a connected relationship graph as parent-child edges.

    Uses a tree structure to ensure connectivity, then optionally adds
    extra edges for more complex schemas.

    Returns:
        List of (parent_idx, child_idx) tuples representing foreign key relationships
    """
    if num_tables < 2:
        return []

    edges = []

    # Build a spanning tree to ensure connectivity
    # Start with table 0 as root
    available_parents = [0]
    remaining_children = list(range(1, num_tables))

    while remaining_children:
        # Pick a random parent from available parents
        parent = rng.choice(available_parents)

        # Pick a random child
        child = remaining_children.pop(0)

        # Add edge
        edges.append((parent, child))

        # Child can now be a parent
        available_parents.append(child)

    # Optionally add extra edges for more complex schemas (not too many)
    # For 4+ tables, add 0-1 extra edges
    if num_tables >= 4:
        num_extra = rng.randint(0, 1)
        for _ in range(num_extra):
            # Pick two different tables
            t1 = rng.randint(0, num_tables - 1)
            t2 = rng.randint(0, num_tables - 1)
            if t1 != t2:
                # Ensure parent comes before child (arbitrary but consistent)
                parent, child = min(t1, t2), max(t1, t2)
                # Only add if not already present
                if (parent, child) not in edges:
                    edges.append((parent, child))

    return edges


def _generate_row_counts(num_tables: int, rng: random.Random) -> List[int]:
    """
    Generate varied row counts for tables.

    Creates a distribution with small, medium, and large tables.
    Row counts range from ~1K to ~1M with logarithmic distribution.
    """
    row_counts = []

    # Generate row counts with logarithmic distribution
    # This creates realistic variation in table sizes
    for _ in range(num_tables):
        # Random exponent between 3.0 (1K) and 6.0 (1M)
        exponent = rng.uniform(3.0, 6.0)
        row_count = int(10 ** exponent)

        # Round to nice numbers
        if row_count < 10000:
            row_count = (row_count // 1000) * 1000
        elif row_count < 100000:
            row_count = (row_count // 10000) * 10000
        else:
            row_count = (row_count // 100000) * 100000

        # Ensure minimum of 1000 rows
        row_count = max(1000, row_count)

        row_counts.append(row_count)

    return row_counts


def _generate_selectivities(edges: List[Tuple[int, int]],
                           table_names: List[str],
                           rng: random.Random) -> Dict[Tuple[str, str], float]:
    """
    Generate join selectivity estimates for each relationship.

    Selectivity represents the fraction of the Cartesian product
    that appears in the join result.

    Values range from 0.0001 to 0.1 (0.01% to 10%).
    """
    selectivities = {}

    for parent_idx, child_idx in edges:
        parent_name = table_names[parent_idx]
        child_name = table_names[child_idx]

        # Canonical ordering: alphabetically sorted
        pair = tuple(sorted([parent_name, child_name]))

        # Generate selectivity with log-uniform distribution
        # Most joins are selective (small values), some are less selective
        log_sel = rng.uniform(-4, -1)  # 10^-4 to 10^-1
        selectivity = 10 ** log_sel

        selectivities[pair] = selectivity

    return selectivities


def generate_database(num_tables: int, seed: int,
                      indexed: bool = False) -> Database:
    """
    Generate a synthetic relational database with deterministic schema and statistics.

    Args:
        num_tables: Number of tables (must be between 3 and 6)
        seed: Random seed for reproducibility
        indexed: Add a seeded subset of foreign-key indexes when true.

    Returns:
        Database object with tables, relationships, and statistics

    Raises:
        ValueError: If num_tables is not between 3 and 6
    """
    # Validate input
    if not (3 <= num_tables <= 6):
        raise ValueError(f"num_tables must be between 3 and 6, got {num_tables}")

    # Create seeded random generator
    rng = random.Random(seed)

    # Generate table names
    table_names = _generate_table_names(num_tables, rng)

    # Generate connected relationship graph
    edges = _generate_connected_schema(num_tables, rng)

    # Generate row counts
    row_counts = _generate_row_counts(num_tables, rng)

    # Generate selectivities
    selectivities = _generate_selectivities(edges, table_names, rng)

    # Build tables
    tables = []
    for idx, name in enumerate(table_names):
        # Primary key column
        pk_name = f"{name.lower()}_id"
        columns = [Column(name=pk_name, data_type="INTEGER")]

        # Add a couple of ordinary attributes
        num_attrs = rng.randint(1, 3)
        attr_names = ["name", "value", "date", "amount", "status", "type", "code"]
        rng.shuffle(attr_names)
        for attr_name in attr_names[:num_attrs]:
            columns.append(Column(name=attr_name, data_type="INTEGER"))

        # Find foreign keys for this table (where this table is the child)
        foreign_keys = []
        for parent_idx, child_idx in edges:
            if child_idx == idx:
                parent_name = table_names[parent_idx]
                fk_column = f"{parent_name.lower()}_id"
                fk_pk = f"{parent_name.lower()}_id"

                foreign_keys.append(ForeignKey(
                    column=fk_column,
                    references_table=parent_name,
                    references_column=fk_pk
                ))

                # Add foreign key column if not already present
                if not any(col.name == fk_column for col in columns):
                    columns.append(Column(name=fk_column, data_type="INTEGER"))

        # Create table
        table = Table(
            name=name,
            columns=columns,
            primary_key=pk_name,
            row_count=row_counts[idx],
            foreign_keys=foreign_keys,
            indexed_columns=(
                [fk.column for fk in foreign_keys if rng.random() < 0.5]
                if indexed else []
            )
        )
        tables.append(table)

    # Create database
    db = Database(
        tables=tables,
        join_selectivities=selectivities,
        seed=seed
    )

    # Validate the generated database
    _validate_database(db)

    return db


def _validate_database(db: Database) -> None:
    """
    Validate that the generated database satisfies all invariants.

    Raises:
        AssertionError: If any invariant is violated
    """
    # 1. Number of tables is between 3 and 6
    assert 3 <= len(db.tables) <= 6, f"Expected 3-6 tables, got {len(db.tables)}"

    # 2. Every table has exactly one primary key
    for table in db.tables:
        assert table.primary_key, f"Table {table.name} has no primary key"
        # Primary key should be in columns
        pk_columns = [col for col in table.columns if col.name == table.primary_key]
        assert len(pk_columns) == 1, f"Table {table.name} primary key not in columns or duplicated"

    # 3. Primary key names are unique within a table (handled by column list)
    for table in db.tables:
        col_names = [col.name for col in table.columns]
        assert len(col_names) == len(set(col_names)), f"Table {table.name} has duplicate column names"

    # 4-6. Every foreign key references an existing table and column
    table_names = {t.name for t in db.tables}
    for table in db.tables:
        for fk in table.foreign_keys:
            assert fk.references_table in table_names, \
                f"Foreign key in {table.name} references non-existent table {fk.references_table}"

            # Find referenced table and check column exists
            ref_table = db.get_table(fk.references_table)
            ref_col_names = [col.name for col in ref_table.columns]
            assert fk.references_column in ref_col_names, \
                f"Foreign key in {table.name} references non-existent column {fk.references_column} in {fk.references_table}"

    # 7. Every row count is positive
    for table in db.tables:
        assert table.row_count > 0, f"Table {table.name} has non-positive row count"

    # 8. Every selectivity satisfies 0 < selectivity <= 1
    for (t1, t2), sel in db.join_selectivities.items():
        assert 0 < sel <= 1, f"Selectivity for {t1}-{t2} is {sel}, must be in (0, 1]"

    # 9. The relationship graph is connected
    _validate_connectivity(db)


def _validate_connectivity(db: Database) -> None:
    """
    Validate that the relationship graph is connected.

    Uses BFS to check that all tables are reachable from table 0.
    """
    if len(db.tables) <= 1:
        return  # Single table or empty is trivially connected

    # Build adjacency list
    adjacency = {table.name: set() for table in db.tables}

    for table in db.tables:
        for fk in table.foreign_keys:
            # Add bidirectional edges
            adjacency[table.name].add(fk.references_table)
            adjacency[fk.references_table].add(table.name)

    # BFS from first table
    visited = set()
    queue = [db.tables[0].name]
    visited.add(db.tables[0].name)

    while queue:
        current = queue.pop(0)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Check all tables were visited
    assert len(visited) == len(db.tables), \
        f"Relationship graph is not connected: visited {len(visited)}/{len(db.tables)} tables"


def main():
    """Demonstration: Generate and display a 4-table database."""
    print("=" * 60)
    print("MILESTONE 2 DEMONSTRATION")
    print("Synthetic Database Generator")
    print("=" * 60)
    print()

    # Generate a 4-table database with seed 42
    db = generate_database(num_tables=4, seed=42)
    print(db)
    print()

    # Demonstrate reproducibility
    print("=" * 60)
    print("REPRODUCIBILITY TEST")
    print("=" * 60)
    print()

    db_copy = generate_database(num_tables=4, seed=42)

    # Check that key properties match
    same_tables = [t1.name == t2.name and t1.row_count == t2.row_count
                   for t1, t2 in zip(db.tables, db_copy.tables)]

    print(f"Same seed (42) produces identical database: {all(same_tables)}")
    print()

    # Generate with different seed
    db_different = generate_database(num_tables=4, seed=123)
    different_tables = any(t1.row_count != t2.row_count
                          for t1, t2 in zip(db.tables, db_different.tables))

    print(f"Different seed (123) produces different database: {different_tables}")
    print()


if __name__ == "__main__":
    main()
