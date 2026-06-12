# Search all jars for a class/package
OUTPUT="/home/win-se-mlcuda01/Desktop/projects/eclipse-m2dl/plugins/jars_listing.txt"
: > "$OUTPUT"
find /home/win-se-mlcuda01/Desktop/projects/eclipse-m2dl/plugins/deeplearning4j.rl.libs -type f -name "*.jar" -print0 | \
while IFS= read -r -d '' jar; do
  {
    echo "File: $(basename "$jar")"
    jar tf "$jar"
    echo
  } >> "$OUTPUT"
done
