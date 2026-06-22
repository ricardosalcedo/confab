import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecs_patterns from "aws-cdk-lib/aws-ecs-patterns";
import * as elasticache from "aws-cdk-lib/aws-elasticache";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as logs from "aws-cdk-lib/aws-logs";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

export class ConfabStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // --- VPC ---
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2,
      natGateways: 1,
    });

    // --- DynamoDB: Usage metering ---
    const usageTable = new dynamodb.Table(this, "UsageTable", {
      partitionKey: { name: "api_key", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "timestamp", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      timeToLiveAttribute: "ttl",
    });

    // --- ElastiCache Redis: Response cache ---
    const cacheSecurityGroup = new ec2.SecurityGroup(this, "CacheSG", {
      vpc,
      description: "Allow Fargate to access Redis",
    });

    const cacheSubnetGroup = new elasticache.CfnSubnetGroup(
      this,
      "CacheSubnetGroup",
      {
        description: "Confab Redis subnet group",
        subnetIds: vpc.privateSubnets.map((s) => s.subnetId),
      }
    );

    const redis = new elasticache.CfnCacheCluster(this, "Redis", {
      cacheNodeType: "cache.t3.micro",
      engine: "redis",
      numCacheNodes: 1,
      vpcSecurityGroupIds: [cacheSecurityGroup.securityGroupId],
      cacheSubnetGroupName: cacheSubnetGroup.ref,
    });

    // --- Secrets: API keys ---
    const apiKeySecret = new secretsmanager.Secret(this, "ApiKeySecret", {
      secretName: "confab/api-keys",
      description: "LLM provider API keys for confab proxy",
    });

    // --- ECS Cluster ---
    const cluster = new ecs.Cluster(this, "Cluster", {
      vpc,
      containerInsights: true,
    });

    // --- Fargate Service with ALB ---
    const service = new ecs_patterns.ApplicationLoadBalancedFargateService(
      this,
      "Service",
      {
        cluster,
        cpu: 256,
        memoryLimitMiB: 512,
        desiredCount: 1,
        taskImageOptions: {
          image: ecs.ContainerImage.fromAsset(".."),
          containerPort: 8080,
          environment: {
            PORT: "8080",
            CONFAB_SAMPLES: "3",
            CONFAB_SCORING: "fast",
            CONFAB_PROVIDER: "auto",
            REDIS_HOST: redis.attrRedisEndpointAddress,
            REDIS_PORT: redis.attrRedisEndpointPort,
            USAGE_TABLE: usageTable.tableName,
          },
          secrets: {
            OPENAI_API_KEY: ecs.Secret.fromSecretsManager(
              apiKeySecret,
              "OPENAI_API_KEY"
            ),
          },
          logDriver: ecs.LogDrivers.awsLogs({
            streamPrefix: "confab",
            logRetention: logs.RetentionDays.TWO_WEEKS,
          }),
        },
        publicLoadBalancer: true,
      }
    );

    // Configure health check on target group
    service.targetGroup.configureHealthCheck({
      path: "/health",
      interval: cdk.Duration.seconds(30),
    });

    // Allow Fargate → Redis
    cacheSecurityGroup.addIngressRule(
      service.service.connections.securityGroups[0],
      ec2.Port.tcp(6379),
      "Fargate to Redis"
    );

    // Grant Fargate → DynamoDB
    usageTable.grantReadWriteData(service.taskDefinition.taskRole);

    // Auto-scaling
    const scaling = service.service.autoScaleTaskCount({
      minCapacity: 1,
      maxCapacity: 10,
    });
    scaling.scaleOnCpuUtilization("CpuScaling", {
      targetUtilizationPercent: 70,
    });
    scaling.scaleOnRequestCount("RequestScaling", {
      requestsPerTarget: 500,
      targetGroup: service.targetGroup,
    });

    // --- Outputs ---
    new cdk.CfnOutput(this, "ProxyUrl", {
      value: `http://${service.loadBalancer.loadBalancerDnsName}/v1/chat/completions`,
      description: "Confab proxy endpoint",
    });

    new cdk.CfnOutput(this, "HealthUrl", {
      value: `http://${service.loadBalancer.loadBalancerDnsName}/health`,
      description: "Health check endpoint",
    });
  }
}
