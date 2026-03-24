## command to print file content of this project

```bash
find . -type f -not -path '*/\.*' -not -name 'flake.lock' -not -name '*.jpg' -print0 \
  | while IFS= read -r -d '' file; do
      echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "┃  $file"
      echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo ""
      cat "$file"
      echo ""
      echo ""
    done
```