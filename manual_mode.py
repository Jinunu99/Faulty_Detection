from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QCheckBox, QPlainTextEdit
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize

def create_layout_manual(parent):
    """
    수동 모드 레이아웃을 생성합니다. parent는 MyWindow 객체입니다.
    """
    widget = QWidget()
    layout = QVBoxLayout(widget)
    
    # 상단 수평 레이아웃
    top_layout = QHBoxLayout()
    top_origin_lbl = QLabel("검사 이미지")
    top_origin_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    top_det_lbl = QLabel("예측 : EfficientDet D0")
    top_det_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter) 
    
    top_ssd_lbl = QLabel("예측 : SSD Mobilenet V2")
    top_ssd_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    top_layout.addWidget(top_origin_lbl)
    top_layout.addWidget(top_det_lbl)
    top_layout.addWidget(top_ssd_lbl)
    top_layout.setSpacing(10)
    top_layout.setContentsMargins(20, 20, 20, 0)
    
    # 중간 수평 레이아웃
    middle_layout = QGridLayout()
    parent.manual_prev_button = QPushButton()
    parent.manual_prev_button.setIcon(QIcon('./widgetimages/prev_arrow.png'))
    parent.manual_prev_button.setIconSize(QSize(40, 40))
    parent.manual_prev_button.setFixedSize(40, 40)
    parent.manual_prev_button.clicked.connect(parent.load_prev_image)

    parent.manual_next_button = QPushButton()
    parent.manual_next_button.setIcon(QIcon('./widgetimages/next_arrow.png'))
    parent.manual_next_button.setIconSize(QSize(40, 40))
    parent.manual_next_button.setFixedSize(40, 40)
    parent.manual_next_button.clicked.connect(parent.load_next_image)

    parent.manual_left_image = QLabel()
    parent.manual_left_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    parent.manual_left_image.setFixedSize(420, 420)
    parent.set_default_image(parent.manual_left_image)
    
    parent.manual_middle_image = QLabel()
    parent.manual_middle_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    parent.manual_middle_image.setFixedSize(420, 420)
    parent.set_default_image(parent.manual_middle_image)
    
    parent.manual_right_image = QLabel()
    parent.manual_right_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    parent.manual_right_image.setFixedSize(420, 420)
    parent.set_default_image(parent.manual_right_image)

    parent.manual_path_text = QLabel("현재 이미지 경로 표시")
    parent.manual_path_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # 이미지 이동 버튼 추가
    middle_layout.addWidget(parent.manual_prev_button, 0, 0)
    middle_layout.addWidget(parent.manual_left_image, 0, 1)
    middle_layout.addWidget(parent.manual_middle_image, 0, 2)
    middle_layout.addWidget(parent.manual_right_image, 0, 3)
    middle_layout.addWidget(parent.manual_next_button, 0, 4)
    middle_layout.addWidget(parent.manual_path_text, 1, 0, 1, 2)
    middle_layout.setSpacing(10)
    middle_layout.setContentsMargins(10, 0, 10, 0)

    # 하단 수평 레이아웃
    bottom_layout = QHBoxLayout()
    effi_log_layout = QVBoxLayout()
    label_effi_log = QLabel("EfficientDet Log")
    label_effi_log.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    parent.manual_log_plaintext_left = QPlainTextEdit()
    parent.manual_log_plaintext_left.setReadOnly(True)
    effi_log_layout.addWidget(label_effi_log)
    effi_log_layout.addWidget(parent.manual_log_plaintext_left)

    ssd_log_layout = QVBoxLayout()
    label_ssd_log = QLabel("SSD Log")
    label_ssd_log.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    parent.manual_log_plaintext_right = QPlainTextEdit()
    parent.manual_log_plaintext_right.setReadOnly(True)
    ssd_log_layout.addWidget(label_ssd_log)
    ssd_log_layout.addWidget(parent.manual_log_plaintext_right)

    logs_layout = QHBoxLayout()
    logs_layout.addLayout(effi_log_layout, 1)
    logs_layout.addLayout(ssd_log_layout, 1)

    middle_bottom_layout = QVBoxLayout()
    checkbox_texts = ['GrayScale Conversion', 'Gaussian blur', 'Canny edge', 'Specular reflection', 'Background Removal']
    parent.checkboxes = [QCheckBox(text) for text in checkbox_texts]
    for checkbox in parent.checkboxes:
        middle_bottom_layout.addWidget(checkbox)
    middle_bottom_layout.setSpacing(10)
    middle_bottom_layout.setContentsMargins(5, 5, 5, 5)
    middle_bottom_layout.addStretch(1)

    right_bottom_layout = QVBoxLayout()
    process_buttons_texts = ['작업 폴더 선택', '예측 시작', '선택된 영상 처리']
    parent.manual_process_buttons = [QPushButton(text) for text in process_buttons_texts]
    for process_button in parent.manual_process_buttons:
        right_bottom_layout.addWidget(process_button)
    parent.manual_process_buttons[0].clicked.connect(parent.select_folder)
    parent.manual_process_buttons[1].clicked.connect(parent.predict_and_compare)
    parent.manual_process_buttons[2].clicked.connect(parent.cv_process)
    right_bottom_layout.setSpacing(10)
    right_bottom_layout.setContentsMargins(5, 5, 5, 5)
    right_bottom_layout.addStretch(1)

    bottom_layout.addLayout(logs_layout, 4)
    bottom_layout.addLayout(middle_bottom_layout, 1)
    bottom_layout.addLayout(right_bottom_layout, 1)
    layout.addLayout(top_layout, 0)
    layout.addLayout(middle_layout, 0)
    layout.addLayout(bottom_layout, 0)
    return widget
