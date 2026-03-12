resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_tasks
  min_capacity       = var.min_tasks
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# Target tracking: keep average CPU at 60%
# Scale OUT when CPU > 60% (new task added within 60s)
# Scale IN  when CPU drops well below 60% (task removed after 300s cooldown)
resource "aws_appautoscaling_policy" "cpu_tracking" {
  name               = "${var.app_name}-cpu-target-tracking"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 60.0
    scale_in_cooldown  = 300  # 5 min — avoid flapping
    scale_out_cooldown = 60   # 1 min — respond quickly to spikes
  }
}
