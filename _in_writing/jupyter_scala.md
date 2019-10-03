
Install scala

```
sudo apt-get install scala
```

Check scala

```
Welcome to Scala version 2.11.6 (OpenJDK 64-Bit Server VM, Java 1.8.0_191).
Type in expressions to have them evaluated.
Type :help for more information.

scala>
```

Install sbt

https://www.scala-sbt.org/1.0/docs/Installing-sbt-on-Linux.html

```
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install sbt
```

Git clone scala package

```
git clone https://github.com/alexarchambault/jupyter-scala.git
```

Build the package

```
cd jupyter-scala
sbt cli/packArchive
```

