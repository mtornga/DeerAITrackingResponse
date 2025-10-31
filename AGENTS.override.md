Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.

For AWS this command seems to work for auth and accessing s3: set -a; source .env; set +a; unset AWS_PROFILE AWS_SESSION_TOKEN; aws s3 ls s3://wildlife-ugv/models/yolo/