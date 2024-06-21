//parametros iniciales para crear el proyecto Terraform con su versión más actualizada
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.54.1"
    }
  }
}

//proveedor aws y zona
provider "aws" {
  region = "us-east-1"
}