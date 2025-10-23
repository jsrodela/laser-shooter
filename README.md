# Laser Shooter

1. Print target paper. (`/target.png`)

2. Setup

```sh
$ uv venv
$ source .venv/bin/activate
$ pip3 install -r requirements.txt
```

3. Adjust white threshold to perceive only red dot from laser.

```sh
$ python red_difference.py
```

4. Once you get the appropriate white threshold value, run main script.

```sh
$ python main.py
```
