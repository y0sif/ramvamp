#!/usr/bin/env bash
# Docs check (CI runs it first): keeps markdown to the allowed paths, CLAUDE.md files to one
# import, private data out of every file, history and codenames out of the user docs, and each
# doc under its line budget. Prints one `path:line: rule: excerpt` line per violation, then a
# count, and exits 1 when there is any. Untracked files that are not ignored are checked too,
# so an internal file fails here before it is committed.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Git with paths printed as they are: a quoted non-ASCII name would match no rule.
git() { command git -c core.quotePath=false "$@"; }

violations=0

# report PATH LINE RULE EXCERPT
report() {
  local excerpt=$4
  excerpt=${excerpt#"${excerpt%%[![:space:]]*}"}
  printf '%s:%s: %s: %s\n' "$1" "$2" "$3" "${excerpt:0:120}"
  violations=$((violations + 1))
}

# report_hits RULE, reading `path:line:text` lines (grep -nH, git grep -n) from stdin
report_hits() {
  local hit path line
  while IFS= read -r hit; do
    path=${hit%%:*}
    hit=${hit#*:}
    line=${hit%%:*}
    report "$path" "$line" "$1" "${hit#*:}"
  done
}

# Tracked files and untracked ones that are not ignored, as long as they exist on disk. File
# lists are NUL-separated from here on, so every name is read whole.
mapfile -d '' files < <(
  git ls-files -z --cached --others --exclude-standard | sort -zu | while IFS= read -r -d '' f; do
    if [[ -f $f ]]; then printf '%s\0' "$f"; fi
  done
)

# matching REGEX: the files whose path matches the extended regex, NUL-separated
matching() {
  local f
  for f in "${files[@]}"; do
    if [[ $f =~ $1 ]]; then printf '%s\0' "$f"; fi
  done
}

# grep_in PATH-REGEX RULE GREP-OPTIONS...: report the matching lines of the files whose path
# matches PATH-REGEX
grep_in() {
  local scope=$1 rule=$2 targets
  shift 2
  mapfile -d '' targets < <(matching "$scope")
  if ((${#targets[@]})); then
    report_hits "$rule" < <(grep -nH "$@" -- "${targets[@]}")
  fi
}

# 1. Markdown lives only at these paths. A new docs page is named here and gets its
#    `!/docs/<page>.md` line in .gitignore. Internal files go under .claude/, never in the repo.
allowed='^(README|CONTRIBUTING|SECURITY)\.md$|^(.+/)?(AGENTS|CLAUDE)\.md$'
allowed+='|^docs/(architecture|benchmarks|comparison|configuration|experiments|install|roadmap|server)\.md$'
allowed+='|^\.github/(.+/)?[^/]+\.md$'
while IFS= read -r -d '' f; do
  [[ $f =~ $allowed ]] || report "$f" 1 allowlist "not an allowed docs path; internal files go under .claude/"
done < <(matching '\.md$')

# 2. A CLAUDE.md holds only `@AGENTS.md`: the guide itself is AGENTS.md.
only_import='^[[:space:]]*@AGENTS\.md[[:space:]]*$'
while IFS= read -r -d '' f; do
  report_hits claude-md < <(grep -nHv -e '^[[:space:]]*$' -e "$only_import" -- "$f")
  grep -q -e "$only_import" -- "$f" || report "$f" 1 claude-md "missing @AGENTS.md"
done < <(matching '(^|/)CLAUDE\.md$')

# 3. Private data, in every text file.
# Home directories other than the placeholders a test fixture (/home/u) and a systemd example
# (/home/you) use.
while IFS= read -r hit; do
  [[ ${hit##*:} =~ ^/home/(u|you)$ ]] || report_hits private-path <<<"$hit"
done < <(git grep -nIoE --untracked '/home/[a-z][A-Za-z0-9_.-]*')
report_hits private-link < <(git grep -nIE --untracked 'claude\.ai/(code/session|chat/)')
# Email addresses. Allowed: noreply ones, reserved example domains, and names that only look
# like an address: systemd template units (user@1000.service) and SSH remotes (git@github.com).
email_ok='noreply|@example\.(com|org|net)$|@[^@]+\.service$|^git@'
while IFS= read -r hit; do
  [[ ${hit#*:*:} =~ $email_ok ]] || report_hits private-email <<<"$hit"
done < <(git grep -nIoE --untracked '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')

# 4. Wording. Every markdown file: no em or en dashes. Every docs page: no fix history (it
#    goes in the commit). The user docs: no test provenance and no internal codenames. The
#    design and measurement pages (architecture, benchmarks, comparison, experiments, roadmap)
#    record measurements and may cite EXP ids.
docs_pages='^(README|CONTRIBUTING|SECURITY)\.md$|^docs/[^/]+\.md$'
user_docs='^(README|CONTRIBUTING|SECURITY)\.md$|^docs/(install|configuration|server)\.md$'
grep_in '\.md$' dash -F -e $'\xe2\x80\x94' -e $'\xe2\x80\x93'
grep_in "$docs_pages" history -iE '\b(since|until|in|see|fixes|closes|issue|pr) #[0-9]+|\(#[0-9]+\)|as in [0-9]+\.[0-9]+\.[0-9]+'
# "measured on" only before a/my/our/his/this, the forms that name the maintainer's machine.
grep_in "$user_docs" provenance -iE '\(measured\)|measured on (a|my|our|his|this) |tested against|tested on [0-9]|mock model'
grep_in "$user_docs" codename -iE '\bEXP-[0-9]+|\b(phase|round) [0-9]+\b|\bthe author\b'

# 5. Line budgets. Cut before adding. docs/configuration.md holds every flag, variable and
#    profile key, so it gets the hub budget.
# budget FILE LIMIT
budget() {
  local n
  n=$(wc -l <"$1")
  ((n <= $2)) || report "$1" $(($2 + 1)) budget "$n lines, the limit is $2"
}
while IFS= read -r -d '' f; do
  case $f in
  README.md) budget "$f" 220 ;;
  AGENTS.md) budget "$f" 150 ;;
  */AGENTS.md) budget "$f" 60 ;;
  CONTRIBUTING.md | SECURITY.md) budget "$f" 300 ;;
  docs/configuration.md) budget "$f" 500 ;;
  docs/*.md) budget "$f" 300 ;;
  esac
done < <(matching '^((README|CONTRIBUTING|SECURITY)\.md|(.+/)?AGENTS\.md|docs/[^/]+\.md)$')

if ((violations)); then
  echo "check-docs: $violations violation(s)"
  exit 1
fi
echo "check-docs: ok"
