variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Application name used as prefix for all resources"
  type        = string
  default     = "rag-app"
}

variable "min_tasks" {
  description = "Minimum number of ECS Fargate tasks"
  type        = number
  default     = 1
}

variable "max_tasks" {
  description = "Maximum number of ECS Fargate tasks"
  type        = number
  default     = 5
}

variable "task_cpu" {
  description = "CPU units for ECS task (512 = 0.5 vCPU)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memory in MiB for ECS task"
  type        = number
  default     = 1024
}

variable "container_port" {
  description = "Port Streamlit listens on inside the container"
  type        = number
  default     = 8501
}
