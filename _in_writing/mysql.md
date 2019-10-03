```
apt-get install mysql-server
```

root 계정의 비밀 번호 입력

설치된 mysql components 확인하기

```
dpkg --list | grep mysql
```

```
ii  mysql-client-5.7                           5.7.26-0ubuntu0.16.04.1                                     amd64        MySQL database client binaries
ii  mysql-client-core-5.7                      5.7.26-0ubuntu0.16.04.1                                     amd64        MySQL database core client binaries
ii  mysql-common                               5.7.26-0ubuntu0.16.04.1                                     all          MySQL database common files, e.g. /etc/mysql/my.cnf
ii  mysql-server                               5.7.26-0ubuntu0.16.04.1                                     all          MySQL database server (metapackage depending on the latest version)
ii  mysql-server-5.7                           5.7.26-0ubuntu0.16.04.1                                     amd64        MySQL database server binaries and system database setup
ii  mysql-server-core-5.7                      5.7.26-0ubuntu0.16.04.1                                     amd64        MySQL database server binaries
```

```
$ service mysql start
$ service mysql stop
```

USERID 로 접속하기

```
mysql -u USREID -p
```

설정파일

```
/etc/mysql/mysql.conf.d/mysqld.cnf
```

기본값
```
user            = mysql
pid-file        = /var/run/mysqld/mysqld.pid
socket          = /var/run/mysqld/mysqld.sock
port            = 3306
basedir         = /usr
datadir         = /var/lib/mysql
tmpdir          = /tmp
lc-messages-dir = /usr/share/mysql
...
bind-address            = 127.0.0.1
```

