!eval "$(ssh-agent -s)"
!ssh-add /root/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts
!ssh -T git@github.com