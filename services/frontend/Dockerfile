FROM nginx:1.27-alpine

# Свой конфиг вместо стандартного
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Статические файлы страницы
COPY public/ /usr/share/nginx/html/

EXPOSE 80
