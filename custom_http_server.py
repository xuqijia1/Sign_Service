# custom_http_server.py - HTTP服务器
import json
import os
import logging
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

from SharedData import shared_data, DetectionBox, SignResult, get_sign_type

logger = logging.getLogger(__name__)

class CustomHTTPRequestHandler(BaseHTTPRequestHandler):
    """自定义HTTP请求处理器"""

    # 高频接口，不记录每次请求日志
    _quiet_paths = {'/api/sign_boxes', '/api/sign_results'}

    def log_message(self, format, *args):
        """重写日志方法，高频轮询接口不刷屏"""
        parsed = urlparse(self.path)
        if parsed.path in self._quiet_paths:
            return
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """发送JSON响应"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_error_response(self, message: str, status_code: int = 400):
        """发送错误响应"""
        self._send_json_response({'error': message}, status_code)

    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        try:
            if path == '/' or path == '/index.html':
                self._handle_index()
            elif path == '/api/Start':
                self._handle_start(query_params)
            elif path == '/api/Stop':
                self._handle_stop(query_params)
            elif path == '/api/sign_boxes':
                self._handle_sign_boxes()
            elif path == '/api/sign_results':
                self._handle_sign_results()
            elif path == '/api/status':
                self._handle_status()
            else:
                self._send_error_response('Not Found', 404)
        except Exception as e:
            logger.error(f"处理GET请求异常: {e}")
            self._send_error_response(str(e), 500)

    def do_POST(self):
        """处理POST请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        try:
            # 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            data = json.loads(body) if body else {}

            if path == '/sign_return':
                self._handle_sign_return(data)
            elif path == '/api/set_config':
                self._handle_set_config(data)
            else:
                self._send_error_response('Not Found', 404)
        except json.JSONDecodeError as e:
            self._send_error_response(f'Invalid JSON: {e}', 400)
        except Exception as e:
            logger.error(f"处理POST请求异常: {e}")
            self._send_error_response(str(e), 500)

    def _handle_index(self):
        """处理首页请求"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sign Service - 标志牌识别服务</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                h1 { color: #333; }
                .api-list { background: white; padding: 20px; border-radius: 8px; }
                .api-item { margin: 10px 0; padding: 10px; background: #e8f5e9; border-radius: 4px; }
                .method { color: #2e7d32; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Sign Service - 标志牌识别服务</h1>
            <div class="api-list">
                <h2>API接口</h2>
                <div class="api-item"><span class="method">GET</span> /api/Start?userid=xxx - 开始识别/录制</div>
                <div class="api-item"><span class="method">GET</span> /api/Stop?userid=xxx - 停止识别/录制</div>
                <div class="api-item"><span class="method">GET</span> /api/sign_boxes - 获取检测框数据</div>
                <div class="api-item"><span class="method">GET</span> /api/sign_results - 获取识别结果</div>
                <div class="api-item"><span class="method">GET</span> /api/status - 获取服务状态</div>
                <div class="api-item"><span class="method">POST</span> /sign_return - 接收识别结果（客户端回调）</div>
            </div>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _handle_start(self, query_params: dict):
        """处理开始请求"""
        userid = query_params.get('userid', [''])[0]

        if not userid:
            self._send_error_response('缺少userid参数')
            return

        # 停止之前的码流录制
        vs = shared_data.video_recorder
        if vs and vs.dvpp_decoder is not None and vs.dvpp_decoder.is_recording:
            try:
                vs.dvpp_decoder.stop_record()
            except Exception as e:
                logger.warning(f"停止码流录制异常: {e}")

        # 设置用户ID
        shared_data.current_user_id = userid

        # 重置状态
        shared_data.recognized_signs = {}
        shared_data.latest_boxes = []
        shared_data.latest_result = None
        shared_data.frame_count = 0

        # 启动码流录制
        video_path = ""
        use_stream_rec = (vs is not None and vs.dvpp_decoder is not None and
                          hasattr(vs.dvpp_decoder, 'start_record'))
        if use_stream_rec:
            date_str = datetime.now().strftime("%Y%m%d")
            time_str = datetime.now().strftime("%H%M%S")
            video_dir = os.path.join(vs._video_save_dir, date_str, userid)
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"sign_record_{userid}_{time_str}.mp4")
            try:
                vs.dvpp_decoder.start_record(video_path)
                logger.info(f"PyAV 码流录制启动: {video_path}")
            except Exception as e:
                logger.warning(f"码流录制启动失败({e})")
                video_path = ""
        shared_data.is_recording = True

        # 设置运行状态
        shared_data.set_running(True)

        logger.info(f"开始识别: userid={userid}")

        self._send_json_response({
            'type': 'GET',
            'path': '/api/Start',
            'userid': userid,
            'video_path': video_path,
            'status': 'started'
        })

    def _handle_stop(self, query_params: dict):
        """处理停止请求"""
        userid = query_params.get('userid', [''])[0]

        # 停止录制
        video_path = ""
        vs = shared_data.video_recorder
        use_stream_rec = (vs is not None and vs.dvpp_decoder is not None and
                          vs.dvpp_decoder.is_recording)
        if use_stream_rec:
            try:
                rec_mp4 = vs.dvpp_decoder.stop_record()
                if rec_mp4:
                    video_path = rec_mp4
                logger.info(f"码流录制停止: {video_path}")
            except Exception as e:
                logger.warning(f"码流录制停止异常: {e}")

        # 获取识别结果
        results = shared_data.get_all_results()

        # 重置状态
        shared_data.set_running(False)
        shared_data.is_recording = False
        shared_data.current_user_id = ""

        logger.info(f"停止识别: userid={userid}, video_path={video_path}")

        self._send_json_response({
            'type': 'GET',
            'path': '/api/Stop',
            'userid': userid,
            'video_path': video_path,
            'recognized_signs': results.get('recognized_signs', []),
            'frame_count': results.get('frame_count', 0),
            'status': 'stopped'
        })

    def _handle_sign_boxes(self):
        """处理获取检测框请求"""
        boxes = shared_data.get_boxes()

        data = {
            'boxes': [
                {
                    'X': b.X,
                    'Y': b.Y,
                    'Width': b.Width,
                    'Height': b.Height,
                    'Label': b.Label,
                    'Confidence': b.Confidence
                }
                for b in boxes
            ],
            'count': len(boxes),
            'timestamp': shared_data.frame_count
        }

        self._send_json_response(data)

    def _handle_sign_results(self):
        """处理获取识别结果请求"""
        result = shared_data.get_result()
        all_results = shared_data.get_all_results()

        if result:
            data = {
                'SignType': result.SignType,
                'SignName': result.SignName,
                'IsCorrect': result.IsCorrect,
                'Confidence': result.Confidence,
                'recognized_signs': all_results.get('recognized_signs', []),
                'frame_count': all_results.get('frame_count', 0),
                'is_running': all_results.get('is_running', False)
            }
        else:
            data = {
                'SignType': '',
                'SignName': '',
                'IsCorrect': False,
                'Confidence': 0.0,
                'recognized_signs': [],
                'frame_count': 0,
                'is_running': False
            }

        self._send_json_response(data)

    def _handle_status(self):
        """处理获取状态请求"""
        all_results = shared_data.get_all_results()

        data = {
            'is_running': all_results.get('is_running', False),
            'is_recording': shared_data.is_recording,
            'user_id': all_results.get('user_id', ''),
            'frame_count': all_results.get('frame_count', 0),
            'elapsed_time': all_results.get('elapsed_time', 0),
            'recognized_count': len(all_results.get('recognized_signs', []))
        }

        self._send_json_response(data)

    def _handle_sign_return(self, data: dict):
        """处理识别结果回调（客户端调用）"""
        # 这个接口通常由服务端推送到客户端，但也可以接收客户端的请求
        logger.info(f"收到识别结果回调: {data.get('SignName', '')}")

        self._send_json_response({'success': True})

    def _handle_set_config(self, data: dict):
        """处理设置配置请求"""
        if 'client_url' in data:
            shared_data.client_url = data['client_url']
        if 'push_interval' in data:
            shared_data.push_interval = float(data['push_interval'])

        logger.info(f"配置已更新: {data}")

        self._send_json_response({'success': True, 'config': data})


def run_server(port: int = 8090):
    """运行HTTP服务器"""
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, CustomHTTPRequestHandler)
    logger.info(f"HTTP服务器启动，监听端口: {port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    finally:
        httpd.server_close()
        logger.info("服务器已关闭")
