```
ssh-keygen -t rsa -P ""
```

RSA 는 암호화 방법입니다.

명령어를 입력하면 key 를 저장할 파일 이름을 입력하라는 메시지가 뜹니다. 파일 이름을 입력하면 [파일].pub 파일이 만들어집니다.

```
Enter file in which to save the key
```

scp 
authorized_keys라는 파일을 만든다. (이미 생성되어 있다면 append)

