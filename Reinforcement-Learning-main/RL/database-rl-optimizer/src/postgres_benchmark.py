"""PostgreSQL benchmark integration for real-world validation.

This module connects the RL agent to actual PostgreSQL execution,
comparing learned join orders against PostgreSQL's native optimizer.
"""

import os
import psycopg2
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from statistics import median
import numpy as np

from src.database import Database
from src.query import Query
from src.cost_model import CostModel
from src.environment import JoinOrderEnv


class PostgreSQLBenchmark:
    """Benchmark RL agents against real PostgreSQL execution."""
    
    def __init__(self, connection_params: Dict[str, str] = None):
        """Initialize PostgreSQL benchmark.
        
        Args:
            connection_params: PostgreSQL connection parameters
                              (defaults to environment variables)
        """
        if connection_params is None:
            connection_params = {
                'host': os.getenv('PGHOST', 'localhost'),
                'port': int(os.getenv('PGPORT', '5432')),
                'user': os.getenv('PGUSER', 'postgres'),
                'password': os.getenv('PGPASSWORD', 'postgres'),
                'dbname': os.getenv('PGDATABASE', 'rl_optimizer')
            }
        
        self.connection_params = connection_params
        self.conn = None
        self.results_dir = Path('results/pg_benchmark')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> bool:
        """Establish PostgreSQL connection.
        
        Returns:
            True if connection successful
        """
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            return True
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def close(self):
        """Close PostgreSQL connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute_query(self, sql: str, explain: bool = True, 
                     analyze: bool = False, num_runs: int = 1) -> Dict[str, Any]:
        """Execute query and collect performance metrics.
        
        Args:
            sql: SQL query to execute
            explain: Whether to use EXPLAIN
            analyze: Whether to actually execute (EXPLAIN ANALYZE)
            num_runs: Number of runs for timing (median taken)
            
        Returns:
            Dictionary with execution metrics
        """
        if not self.conn:
            raise RuntimeError("Not connected to PostgreSQL")
        
        cursor = self.conn.cursor()
        results = {
            'sql': sql,
            'explain': explain,
            'analyze': analyze,
            'num_runs': num_runs
        }
        
        try:
            # Warm-up run if analyzing
            if analyze and num_runs > 1:
                cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
                cursor.fetchall()
            
            # Collect runs
            execution_times = []
            planning_times = []
            plans = []
            
            for _ in range(num_runs):
                if explain:
                    if analyze:
                        query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
                    else:
                        query = f"EXPLAIN (FORMAT JSON) {sql}"
                else:
                    query = sql
                
                start = time.time()
                cursor.execute(query)
                result = cursor.fetchall()
                elapsed = time.time() - start
                
                if explain:
                    plan_json = result[0][0][0]
                    plans.append(plan_json)
                    
                    if analyze:
                        execution_times.append(plan_json.get('Execution Time', 0))
                        planning_times.append(plan_json.get('Planning Time', 0))
                else:
                    execution_times.append(elapsed * 1000)  # Convert to ms
            
            # Store metrics
            if execution_times:
                results['execution_time_ms'] = median(execution_times)
                results['execution_times'] = execution_times
            
            if planning_times:
                results['planning_time_ms'] = median(planning_times)
                results['planning_times'] = planning_times
            
            if plans:
                # Extract plan details from first run
                plan = plans[0]['Plan']
                results['plan'] = plan
                results['total_cost'] = plan.get('Total Cost', 0)
                results['plan_rows'] = plan.get('Plan Rows', 0)
                results['join_order'] = self._extract_join_order(plan)
            
            results['success'] = True
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
        
        finally:
            cursor.close()
            self.conn.rollback()  # Don't commit, keep DB clean
        
        return results
    
    def _extract_join_order(self, plan: Dict) -> List[str]:
        """Extract join order from PostgreSQL plan tree.
        
        Args:
            plan: PostgreSQL plan JSON
            
        Returns:
            List of table names in join order
        """
        tables = []
        
        def traverse(node):
            node_type = node.get('Node Type', '')
            
            # Extract table name from scan nodes
            if 'Scan' in node_type:
                relation = node.get('Relation Name')
                if relation:
                    tables.append(relation)
            
            # Recurse into children
            for child in node.get('Plans', []):
                traverse(child)
        
        traverse(plan)
        return tables
    
    def force_join_order(self, query: Query, join_order: List[str]) -> str:
        """Generate SQL with forced join order.
        
        Args:
            query: Query object
            join_order: Ordered list of table names
            
        Returns:
            SQL with explicit join order
        """
        # Build explicit nested joins
        if len(join_order) < 2:
            return None
        
        # Start with first table
        from_clause = join_order[0]
        joined_tables = {join_order[0]}
        
        # Add each subsequent table with explicit JOIN
        for table in join_order[1:]:
            # Find join predicate
            predicate = None
            for (t1, c1), (t2, c2) in query.join_graph.edges:
                if (t1 in joined_tables and t2 == table) or \
                   (t2 in joined_tables and t1 == table):
                    predicate = f"{t1}.{c1} = {t2}.{c2}"
                    break
            
            if predicate:
                from_clause = f"({from_clause} JOIN {table} ON {predicate})"
                joined_tables.add(table)
            else:
                # Skip if no valid join found
                continue
        
        # Build complete SQL
        select_cols = ", ".join([f"{t}.{t[0]}_*" for t in join_order[:2]])
        sql = f"SELECT {select_cols} FROM {from_clause}"
        
        return sql
    
    def benchmark_agent(self, agent, env: JoinOrderEnv, 
                       num_queries: int = 10) -> Dict[str, Any]:
        """Benchmark agent against PostgreSQL optimizer.
        
        Args:
            agent: RL agent (must have select_action method)
            env: Join order environment
            num_queries: Number of test queries
            
        Returns:
            Benchmark results dictionary
        """
        results = {
            'agent_type': agent.__class__.__name__,
            'num_queries': num_queries,
            'queries': []
        }
        
        for q_id in range(num_queries):
            # Generate query
            query = env.query
            
            # 1. Get PostgreSQL's default plan
            default_sql = self._build_default_sql(query)
            pg_result = self.execute_query(default_sql, explain=True, 
                                          analyze=True, num_runs=3)
            
            # 2. Get agent's join order
            obs, info = env.reset()
            done = False
            agent_order = [env.start_table]
            
            while not done:
                valid_actions = info['valid_actions']
                action = agent.select_action(obs, valid_actions, training=False)
                obs, reward, done, _, info = env.step(action)
                agent_order.append(env.query.tables[action])
            
            # 3. Force agent's order in PostgreSQL
            forced_sql = self.force_join_order(query, agent_order)
            if forced_sql:
                # Disable PostgreSQL's optimizer
                cursor = self.conn.cursor()
                cursor.execute("SET join_collapse_limit = 1")
                cursor.execute("SET from_collapse_limit = 1")
                cursor.close()
                
                agent_result = self.execute_query(forced_sql, explain=True,
                                                 analyze=True, num_runs=3)
                
                # Re-enable optimizer
                cursor = self.conn.cursor()
                cursor.execute("RESET join_collapse_limit")
                cursor.execute("RESET from_collapse_limit")
                cursor.close()
            else:
                agent_result = {'success': False, 'error': 'Invalid join order'}
            
            # Store comparison
            query_result = {
                'query_id': q_id,
                'num_tables': len(query.tables),
                'postgresql': {
                    'execution_time_ms': pg_result.get('execution_time_ms'),
                    'total_cost': pg_result.get('total_cost'),
                    'join_order': pg_result.get('join_order')
                },
                'agent': {
                    'execution_time_ms': agent_result.get('execution_time_ms'),
                    'total_cost': agent_result.get('total_cost'),
                    'join_order': agent_order
                }
            }
            
            # Calculate improvement
            if pg_result.get('execution_time_ms') and agent_result.get('execution_time_ms'):
                improvement = (pg_result['execution_time_ms'] - 
                              agent_result['execution_time_ms']) / \
                             pg_result['execution_time_ms'] * 100
                query_result['improvement_percent'] = improvement
            
            results['queries'].append(query_result)
        
        # Aggregate statistics
        improvements = [q['improvement_percent'] for q in results['queries'] 
                       if 'improvement_percent' in q]
        
        if improvements:
            results['summary'] = {
                'mean_improvement': np.mean(improvements),
                'median_improvement': np.median(improvements),
                'std_improvement': np.std(improvements),
                'wins': sum(1 for i in improvements if i > 0),
                'losses': sum(1 for i in improvements if i < 0),
                'ties': sum(1 for i in improvements if i == 0)
            }
        
        return results
    
    def _build_default_sql(self, query: Query) -> str:
        """Build standard SQL letting PostgreSQL choose join order."""
        tables = query.tables
        select_cols = ", ".join([f"{t}.{t[0]}_*" for t in tables[:2]])
        
        # Natural FROM clause
        from_clause = ", ".join(tables)
        
        # WHERE clause with all join predicates
        where_clauses = []
        for (t1, c1), (t2, c2) in query.join_graph.edges:
            where_clauses.append(f"{t1}.{c1} = {t2}.{c2}")
        
        where_clause = " AND ".join(where_clauses)
        
        sql = f"SELECT {select_cols} FROM {from_clause}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        return sql
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to JSON.
        
        Args:
            results: Results dictionary
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_{results['agent_type']}_{timestamp}.json"
        
        output_path = self.results_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved results to {output_path}")
        
        return output_path


def main():
    """Demo PostgreSQL benchmark."""
    benchmark = PostgreSQLBenchmark()
    
    if not benchmark.connect():
        print("Failed to connect to PostgreSQL")
        print("Make sure Docker container is running: docker compose up -d")
        return
    
    print("✓ Connected to PostgreSQL")
    
    # Test simple query
    test_sql = "SELECT version()"
    result = benchmark.execute_query(test_sql, explain=False)
    
    if result['success']:
        print(f"✓ PostgreSQL test successful")
        print(f"  Execution time: {result['execution_time_ms']:.2f} ms")
    
    # Test EXPLAIN
    test_sql = "SELECT * FROM customer LIMIT 10"
    result = benchmark.execute_query(test_sql, explain=True, analyze=False)
    
    if result['success']:
        print(f"✓ EXPLAIN test successful")
        print(f"  Estimated cost: {result['total_cost']:.2f}")
    
    benchmark.close()
    print("\n✓ Benchmark module ready")


if __name__ == "__main__":
    main()
