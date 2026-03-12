# Secrets Manager entries — values are set via CLI after terraform apply.
# Never put real secret values here; Terraform state would store them in plaintext.
#
# After apply, populate with:
#   aws secretsmanager put-secret-value --secret-id rag-app/openai-api-key --secret-string "sk-..."
#   aws secretsmanager put-secret-value --secret-id rag-app/astra-db-api-endpoint --secret-string "https://..."
#   aws secretsmanager put-secret-value --secret-id rag-app/astra-db-application-token --secret-string "AstraCS:..."

resource "aws_secretsmanager_secret" "openai_api_key" {
  name                    = "${var.app_name}/openai-api-key"
  recovery_window_in_days = 7
  tags                    = { Name = "${var.app_name}-openai-api-key" }
}

resource "aws_secretsmanager_secret_version" "openai_api_key" {
  secret_id     = aws_secretsmanager_secret.openai_api_key.id
  secret_string = "PLACEHOLDER_REPLACE_AFTER_APPLY"

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret" "astra_endpoint" {
  name                    = "${var.app_name}/astra-db-api-endpoint"
  recovery_window_in_days = 7
  tags                    = { Name = "${var.app_name}-astra-endpoint" }
}

resource "aws_secretsmanager_secret_version" "astra_endpoint" {
  secret_id     = aws_secretsmanager_secret.astra_endpoint.id
  secret_string = "PLACEHOLDER_REPLACE_AFTER_APPLY"

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret" "astra_token" {
  name                    = "${var.app_name}/astra-db-application-token"
  recovery_window_in_days = 7
  tags                    = { Name = "${var.app_name}-astra-token" }
}

resource "aws_secretsmanager_secret_version" "astra_token" {
  secret_id     = aws_secretsmanager_secret.astra_token.id
  secret_string = "PLACEHOLDER_REPLACE_AFTER_APPLY"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
