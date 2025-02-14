# Docker Tutorial


# 1. Команда docker run -p 8888:8888 jupyter/scipy-notebook:2c80cf3537ca используется 
# для запуска контейнера с образом Jupyter Notebook, на основе конкретной версии scipy-notebook.
# Вот что означает каждая часть команды:
# - docker run: Эта команда создает и запускает новый контейнер.
# - -p 8888:8888: Этот параметр публикует порт контейнера на хост-машине.
# - 8888:8888 (HOST_PORT:CONTAINER_PORT) означает, 
# что порт 8888 контейнера будет отображен на порт 8888 на хост-машине. 
# Это позволяет вам получать доступ к Jupyter Notebook через веб-браузер, открыв http://localhost:8888.
# - jupyter/scipy-notebook:2c80cf3537ca: Это имя и тег Docker-образа, который будет использоваться для создания контейнера.
#  - jupyter/scipy-notebook: Имя образа Docker, который включает в себя Jupyter Notebook с поддержкой SciPy и всей экосистемы Python.
#  - 2c80cf3537ca: Это идентификатор конкретной версии образа.
# Таким образом, эта команда запускает Jupyter Notebook, который будет доступен на порту 8888 на вашем компьютере.
docker run -p 8888:8888 jupyter/scipy-notebook:2c80cf3537ca

# 2. Команда docker exec -it <mycontainer> bash используется для выполнения интерактивной оболочки Bash в работающем контейнере Docker.
# Вот что означает каждая часть команды:
# - docker exec: Эта команда позволяет выполнять команды внутри запущенного контейнера.
# - -it: Флаги -i и -t указывают Docker, что вы хотите запустить интерактивный терминал. 
# - -i означает, что вы хотите поддерживать стандартный ввод контейнера.
# - -t означает, что вы хотите создать терминал (TTY).
# - <mycontainer>: Это идентификатор (ID) контейнера, в котором вы хотите запустить оболочку.
# - bash: Это команда, которую вы хотите запустить внутри контейнера (в данном случае — оболочка Bash).
# Итак, после выполнения этой команды вы получите интерактивный терминал Bash внутри указанного контейнера.
docker exec -it <mycontainer> bash
docker exec -it -u root <container_name_or_id> /bin/bash (с правами админа)

# 3. Команда docker cp <file> <mycontainer>:<dir> используется для копирования файлов или 
# директорий с вашего локального хост-машины в запущенный Docker-контейнер.
# Вот что означает каждая часть команды:
# - docker cp: Команда Docker для копирования файлов или директорий между локальной файловой системой и файловой системой контейнера.
# - <file>: Путь к файлу или директории, которые вы хотите скопировать с вашей локальной машины.
# - <mycontainer>: Идентификатор или имя контейнера, в который вы хотите скопировать файл.
# - <dir>: Путь в контейнере, куда вы хотите скопировать файл или директорию.
docker cp <file> <mycontainer>:</dir>

# 4. Volume:
# Команда docker run -v <HOST_DIR>:<CONTAINER_DIR> -p <HOST_PORT:CONTAINER_PORT> <image> 
# запускает новый Docker-контейнер с монтированием локальной директории и пробросом порта.
# Вот что означает каждая часть команды:
# - docker run: Команда для запуска нового контейнера.
# - -v <HOST_DIR>:<CONTAINER_DIR>: Опция для монтирования локальной директории (<HOST_DIR>) внутрь контейнера по указанному пути (<CONTAINER_DIR>). 
# Это позволяет вашему контейнеру доступиться к данным или использовать конфигурации с вашего хоста.
# - -p <HOST_PORT:CONTAINER_PORT>: Опция для проброса порта. Указывает, что порт CONTAINER_PORT контейнера будет доступен на порту HOST_PORT вашего хоста.
# - <mycontainer>: Имя или идентификатор образа Docker, на основе которого создается контейнер.
docker run -v <HOST_DIR>:<CONTAINER_DIR> -p <HOST_PORT:CONTAINER_PORT> <image>

# 5. Dockerfile
# FROM - команда используется в Dockerfile для указания базового образа, на основе которого будет создаваться новый образ. 
# Это стартовая точка для сборки вашего контейнера: FROM <базовый образ>
# RUN - выполняет команды в процессе сборки образа. Это позволяет вам устанавливать пакеты, копировать файлы, выполнять скрипты и т.д: RUN <команда на выполнение>
# Команды RUN выполняются во время сборки образа, и результат сохраняется в образе.
# Команды на исполнение после создания Dockerfile
docker build <DOCKERFILE_DIR>
docker run -v <HOST_DIR>:<CONTAINER_DIR> -p <HOST_PORT:CONTAINER_PORT> <image>

# 6. Docker-Compose
# Docker Compose — это инструмент, который упрощает управление многоконтейнерными Docker-приложениями. 
# version: '3' - указывает версию синтаксиса файла Docker-Compose. Версия 3 является одной из самых популярных;
# services: - раздел, в котором описываются все сервисы, входящие в состав многоконтейнерного приложения.
# Каждый сервис определяет один или несколько контейнеров.
# webapp: - имя сервиса, это название для веб-приложения (любое). Используется для ссылки на сервис внутри файла Docker Compose.
# build: - раздел с описанием. То есть там описаны инструкции по сборке образа для данного сервиса (веб-приложения);
# context: - указывает путь к директории контекста сборки. Контекстом сборки может быть текущая директория (.) или любой другой путь, откуда будет загружаться Dockerfile и прочие файлы для сборки образа;
# dockerfile: - указывает путь к Dockerfile, если он не находится в корневой директории контекста сборки
# (dockerfile: Dockerfile - такое обозначение говорит, что Dockerfile находится в директории контекста);
# image: postgres: задаёт образ Docker, который будет использоваться для создания контейнера. В данном случае, используется официальный образ PostgreSQL из Docker Hub;
# restart: always - указывает политику перезапуска контейнера. В данном случае, используется политика always. Значение (always): контейнер будет всегда перезапускаться, если он остановится по любой причине, кроме случаев, когда он был явно остановлен пользователем (например, через docker stop).
# args: - раздел для передачи аргументов сборки в Dockerfile. Эти аргументы могут быть потом использованы в Dockerfile как переменные:

# example #1
<!-- version: '3'
services:
  webapp:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        NODE_ENV: production -->

# example #2
<!-- version: '3'
services:
  jupyter:
    build:
      context: ./
      dockerfile: dockerfile
    volumes:
      - ./:/home/jovyan/work/
    ports:
      - '8888:8888'
  db:
    images: postgres
    restart: always
    environment:
      POSTGRES_PASSWORD: 123 -->

# example #3
<!-- version: '3'
services:
  jupyter:
    build:
      context: ./
      dockerfile: Dockerfile
    volumes:
      - ./jup:/home/jovyan/work
    ports:
      - '8888:8888'
  db_pg:
    image: postgres
    restart: always
    volumes:
      - ./pgdata:/var/lib/postgresql/data
    ports:
      - '5432:5432'
    environment:
      POSTGRES_PASSWORD: 123
volumes:
  pgdata:
  jup: -->

# Команда docker-compose up используется для запуска всех сервисов, указанных в файле docker-compose.yml.
# Если образы, указанные в файле, ещё не собраны или не загружены, docker-compose up сначала попытается собрать или загрузить их, а затем запустит контейнеры.
# По умолчанию, docker-compose up будет выводить журнал работы всех контейнеров в терминал (можно наблюдать логи контейнеров в реальном времени).
# docker-compose up -d - запускает контейнеры в фоновом режиме (detached mode).  
# Это позволит вам закрыть терминал или продолжить использовать его для других команд, в то время как контейнеры будут работать в фоне.
# docker-compose up --build - принудительная пересборка образов перед запуском контейнеров, даже если образы уже существуют.
docker-compose up

# docker-compose down: - это команда, которая останавливает и удаляет все контейнеры
docker-compose down

# 7. Docker-compose + Postgres


# 8. Docker + Git
# git init: - команда для инциализации папки (создание репозитория в папке) для загрузки на Docker репозиторий (тут будут отслеживаться изменения файлов # для загузки в репозиторий);
# .gitignore: - файл с названиями файлов/папок, которые мы проигнорируем для загрузки в репозиторий;
# git add .: - команда для сохранения в git контроля версий (таким образом фиксируем последние обновления), где . означает добавить все объекты;

