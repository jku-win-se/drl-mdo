#!/usr/bin/env bash
# Compile the RL environment server from source into server-executables/build/.
#
# Output:
#   build/classes/   compiled server and case-study classes
#   build/lib/       third-party jars (repo libs plus EMF, Henshin and OCL from Eclipse)
#   build/VERSION    commit and dirty flag the build came from
#
# The result is self-contained, so build/ can be copied to a machine without Eclipse.
#
# Settings, from .env or the environment:
#   JAVA_BIN         path to the java binary (javac must sit next to it)
#   ECLIPSE_PLUGINS  plugins/ folder of an Eclipse Modeling install with Henshin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }

: "${ECLIPSE_PLUGINS:?Set ECLIPSE_PLUGINS in .env to the plugins/ folder of an Eclipse Modeling install}"
[[ -d "$ECLIPSE_PLUGINS" ]] || { echo "Error: ECLIPSE_PLUGINS not found: $ECLIPSE_PLUGINS" >&2; exit 1; }

JAVAC="javac"
if [[ -n "${JAVA_BIN:-}" ]]; then
  JAVAC="$(dirname "$JAVA_BIN")/javac"
fi
command -v "$JAVAC" >/dev/null || { echo "Error: javac not found: $JAVAC" >&2; exit 1; }

API="$ROOT/momot-rl-env-api"
BUILD="$SCRIPT_DIR/build"
TMP="$BUILD.tmp"
rm -rf "$TMP"
mkdir -p "$TMP/classes" "$TMP/lib"

SOURCE_DIRS=(
  "$API/plugins/at.ac.tuwien.big.moea/src"
  "$API/plugins/at.ac.tuwien.big.momot.core/src"
  "$API/examples/org.cra2/src"
  "$API/examples/org.cra2/src-gen"
  "$API/examples/org.scrum/src"
  "$API/examples/org.scrum/src-gen"
  "$API/examples/org.scrum/xtend-gen"
  "$API/examples/at.ac.tuwien.big.momot.examples.stack/src"
  # momot.core imports the encoding strategies of these domains.
  "$API/examples/at.ac.tuwien.big.momot.examples.pacman/src"
  "$API/examples/at.ac.tuwien.big.momot.examples.container/src"
  "$API/examples/at.ac.tuwien.big.momot.examples.refactoring/src"
  "$API/examples/org.learnapi/src"
  "$API/examples/org.learnapi/src-gen"
)

# Jars the plugins' .classpath files list.
REPO_JARS=(
  "$API/plugins/at.ac.tuwien.big.moea/lib/moeaframework-2.13.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/JMetal-4.3.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/commons-cli-1.3.1.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/commons-codec-1.8.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/commons-io-2.4.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/commons-lang3-3.4.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/commons-math3-3.6.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/guava-18.0.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/jcommon-1.0.23.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/jfreechart-1.0.19.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/rsyntaxtextarea.jar"
  "$API/plugins/at.ac.tuwien.big.moea/lib/procbridge-1.1.1.jar"
  "$API/plugins/deeplearning4j.rl.libs/"*.jar
)

# Eclipse bundles by symbolic name; the newest version of each is used.
ECLIPSE_BUNDLES=(
  org.eclipse.core.runtime org.eclipse.core.jobs org.eclipse.core.contenttype
  org.eclipse.core.resources org.eclipse.equinox.common org.eclipse.equinox.registry
  org.eclipse.equinox.preferences org.eclipse.equinox.app org.eclipse.osgi
  org.eclipse.emf.common org.eclipse.emf.ecore org.eclipse.emf.ecore.xmi
  org.eclipse.emf.ecore.change org.eclipse.emf.edit
  org.eclipse.emf.henshin.model org.eclipse.emf.henshin.interpreter
  org.eclipse.emf.henshin.statespace
  org.eclipse.ocl org.eclipse.ocl.common org.eclipse.ocl.ecore lpg.runtime.java
  org.eclipse.xtend.lib org.eclipse.xtend.lib.macro org.eclipse.xtext.xbase.lib
  org.apache.commons.lang javax.annotation org.jsoup
)

for jar in "${REPO_JARS[@]}"; do
  cp "$jar" "$TMP/lib/"
done

for bundle in "${ECLIPSE_BUNDLES[@]}"; do
  jar="$(ls "$ECLIPSE_PLUGINS/${bundle}_"*.jar 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "$jar" ]]; then
    echo "Error: bundle $bundle not found in $ECLIPSE_PLUGINS" >&2
    exit 1
  fi
  cp "$jar" "$TMP/lib/"
  # Bundles can nest jars, e.g. Henshin's nashorn-core, which modern JDKs no longer ship.
  (cd "$TMP/lib" && unzip -qqjo "$jar" 'lib/*.jar' 2>/dev/null || true)
done

CP="$(printf '%s:' "$TMP"/lib/*.jar)"

SOURCES="$TMP/sources.txt"
for dir in "${SOURCE_DIRS[@]}"; do
  [[ -d "$dir" ]] && find "$dir" -name '*.java'
done > "$SOURCES"

echo "Compiling $(wc -l < "$SOURCES") source files with $("$JAVAC" -version 2>&1)..."
"$JAVAC" --release 17 -nowarn -proc:none -encoding UTF-8 -Xlint:none -Xmaxerrs 200 \
  -cp "$CP" -d "$TMP/classes" @"$SOURCES"

# Resources next to the sources (e.g. Xtend-generated data, model files).
for dir in "${SOURCE_DIRS[@]}"; do
  [[ -d "$dir" ]] || continue
  (cd "$dir" && find . -type f ! -name '*.java' ! -name '*.xtend' ! -name '*._trace' ! -name 'desktop.ini' \
    -exec cp --parents {} "$TMP/classes/" \;)
done

# Service registrations (Henshin's JavaScript engine lookup).
mkdir -p "$TMP/classes/META-INF/services"
cp "$API/plugins/at.ac.tuwien.big.momot.core/META-INF/services/"* "$TMP/classes/META-INF/services/"

{
  git -C "$ROOT" rev-parse HEAD
  git -C "$ROOT" status --porcelain -- momot-rl-env-api | grep -q . && echo dirty || echo clean
} > "$TMP/VERSION"

rm -f "$SOURCES"
rm -rf "$BUILD"
mv "$TMP" "$BUILD"
echo "Built $BUILD ($(ls "$BUILD/lib" | wc -l) jars)."
