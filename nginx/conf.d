worker_processes 1;

events { worker_connections 1024; }

http {

    upstream docker-app {
        server app;
    }

    server {
        listen 80;

        location / {
            proxy_pass         http://docker-app;
            proxy_redirect     off;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;
            proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header   X-Forwarded-Host $server_name;
        }
    }

}