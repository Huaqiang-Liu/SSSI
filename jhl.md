### 命令行记忆 
通过ssh连接虚拟机的方法：
```
ssh -p 2222 ltt@localhost
```
### virtio
打开virtiofsd，host的命令：
```
sudo /usr/libexec/virtiofsd   --socket-path=/tmp/vhostqemu   --shared-dir=/home/user/share   --cache=always
```
在开启guest时适配virtio，开启脚本在~/share/start_vm.sh