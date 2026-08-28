#!/usr/bin/env bash
#
# Start/stop/status helper for the project-local Neo4j Community instance.
#
# Neo4j Community 5.26 runs from a tarball inside .neo4j-local/ together with a
# bundled Temurin JDK 21, since Neo4j 5.x only supports Java 17 or 21. Nothing is
# installed system-wide and no sudo is required. Data persists in
# .neo4j-local/neo4j-community-*/data/.
#
# Usage:
#   scripts/neo4j_local.sh start|stop|restart|status|console|cypher|set-password <pw>

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="$PROJECT_ROOT/.neo4j-local"

NEO4J_HOME="$(find "$LOCAL_DIR" -maxdepth 1 -type d -name 'neo4j-community-*' | sort | tail -1)"
JAVA_HOME="$(find "$LOCAL_DIR" -maxdepth 1 -type d -name 'jdk-21*' | sort | tail -1)"

if [[ -z "$NEO4J_HOME" || -z "$JAVA_HOME" ]]; then
    echo "ERROR: Neo4j and/or JDK 21 not found under $LOCAL_DIR" >&2
    echo "See README.md -> 'Neo4j setup' for provisioning instructions." >&2
    exit 1
fi

export NEO4J_HOME JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

case "${1:-status}" in
    start)   exec "$NEO4J_HOME/bin/neo4j" start ;;
    stop)    exec "$NEO4J_HOME/bin/neo4j" stop ;;
    restart) exec "$NEO4J_HOME/bin/neo4j" restart ;;
    status)  exec "$NEO4J_HOME/bin/neo4j" status ;;
    console) exec "$NEO4J_HOME/bin/neo4j" console ;;
    cypher)
        shift
        exec "$NEO4J_HOME/bin/cypher-shell" "$@"
        ;;
    set-password)
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 set-password <password>" >&2
            exit 2
        fi
        exec "$NEO4J_HOME/bin/neo4j-admin" dbms set-initial-password "$2"
        ;;
    *)
        echo "Usage: $0 start|stop|restart|status|console|cypher|set-password <pw>" >&2
        exit 2
        ;;
esac
