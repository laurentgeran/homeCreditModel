mkdir -p ~/.unicorn/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.unicorn/config.toml