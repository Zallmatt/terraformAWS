//se crea el servicio s3 y lo llamamos "ucp-bot-rn"
resource "aws_s3_bucket" "bucket-ucp" {
    bucket = "ucp-bot-rn"

    tags = {
        Name = "Backup storage"
        Environment = "Dev"
    }
}

//se crea el servicio s3 y lo llamamos "ucp-bot-rn"
resource "aws_s3_bucket" "bucket-de-prueba" {
    bucket = "bucketPrueba"

    tags = {
        Name = "Backup storage"
        Environment = "Dev"
    }
}


resource "aws_instance" "instanciaPrueba" {
  ami           = "ami-08a0d1e16fc3f61ea"
  instance_type = "t2.micro"

  tags = {
    Name = "Manager"
  }

  provisioner "local-exec" {
    command = "echo ${aws_instance.controladorBot.public_ip}"
  }
}


//se suben los archivos que utilizará la red neuronal al bucke creado
//se agrega una key y la ruta al archivo
resource "aws_s3_object" "emociones" {
  bucket = aws_s3_bucket.bucket-ucp.id
  key    = "emociones.csv"
  source = "emociones.csv"

  tags = {
    Name = "dataset"
  }
}

resource "aws_s3_object" "recomendaciones" {
  bucket = aws_s3_bucket.bucket-ucp.id
  key    = "recomendaciones.csv"
  source = "recomendaciones.csv"

  tags = {
    Name = "dataset1"
  }
}

resource "aws_s3_object" "interacciones" {
  bucket = aws_s3_bucket.bucket-ucp.id
  key    = "interacciones.csv"
  source = "interacciones.csv"

  tags = {
    Name = "dataset2"
  }
}

//se crea ua instancia EC2 de prueba, que le dimos el nombre Manager.
//local-exec sirve para ejecutar el codigo localmente en la vm que se creó
resource "aws_instance" "controladorBot" {
  ami           = "ami-08a0d1e16fc3f61ea"
  instance_type = "t2.micro"

  tags = {
    Name = "Manager"
  }

  provisioner "local-exec" {
    command = "echo ${aws_instance.controladorBot.public_ip}"
  }
}

