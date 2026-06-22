#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { ConfabStack } from "./confab-stack";

const app = new cdk.App();

new ConfabStack(app, "ConfabStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "us-west-2",
  },
});
