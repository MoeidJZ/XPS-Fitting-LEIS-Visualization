"""
XPS Analysis Tool V4 - True FFT Convolution & High Precision
Installation: pip install PyQt5 numpy matplotlib scipy
Usage: python XPS_GUI.py
"""

import sys
import copy
import re
import numpy as np
from scipy.signal import fftconvolve
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QComboBox, QScrollArea, QFileDialog, QMessageBox,
                             QGridLayout, QGroupBox, QDoubleSpinBox, QShortcut,
                             QDialog, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QFrame, QCheckBox)
from PyQt5.QtGui import QKeySequence, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# Colorblind-safe palette
COLORS = {
    'raw': '#0173B2',
    'background': '#DE8F05',
    'fit': '#029E73',
    'peaks': [
        '#CC78BC', '#CA9161', '#949494', '#ECE133', '#56B4E9', '#E69F00',
        '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999'
    ]
}

LINE_SHAPES = [
    'GL(30)', 'GL(50)', 'GL(70)',
    'SGL(30)', 'SGL(50)', 'SGL(70)',
    'LF(1,1,30)', 'LF(1.5,1,50)', 'LF(2,1,70)'
]

class UndoManager:
    def __init__(self, limit=15):
        self.limit = limit
        self.stack = []
        self.is_undoing = False

    def push_state(self, state):
        if self.is_undoing: return
        self.stack.append(copy.deepcopy(state))
        if len(self.stack) > self.limit:
            self.stack.pop(0)

    def pop_state(self):
        if not self.stack: return None
        self.is_undoing = True
        state = self.stack.pop()
        self.is_undoing = False
        return state

class XPSPlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_spectrum(self, data, region_name, background_subtracted=False, 
                      background_data=None, peaks_data=None, show_peaks=False):
        self.axes.clear()
        
        sorted_data = sorted(data, key=lambda x: x['be'], reverse=True)
        be_values = [d['be'] for d in sorted_data]
        
        if background_subtracted:
             raw_vals = [d.get('countsSubtracted', 0) for d in sorted_data]
        else:
             raw_vals = [d['counts'] for d in sorted_data]
             
        max_val = max(raw_vals) if raw_vals else 1
        if max_val <= 0: max_val = 1
        
        exponent = int(np.floor(np.log10(max_val)))
        if exponent < 0: exponent = 0
        scale_factor = 10**exponent
        
        ylabel_text = f'Counts/s (x$10^{{{exponent}}}$)' if exponent > 0 else 'Counts/s'
        def scale_arr(arr): return [x / scale_factor for x in arr]
        
        if background_subtracted:
            counts = scale_arr([d.get('countsSubtracted', 0) for d in sorted_data])
            self.axes.plot(be_values, counts, 'o', color=COLORS['raw'], markersize=4, label='Data', alpha=0.6)
            
            if show_peaks and peaks_data:
                sum_peaks = peaks_data['sum']
                for idx, peak in enumerate(peaks_data['peaks']):
                    peak_be = [p['be'] for p in peak]
                    peak_counts = scale_arr([p['counts'] for p in peak])
                    color = COLORS['peaks'][idx % len(COLORS['peaks'])]
                    self.axes.fill_between(peak_be, 0, peak_counts, alpha=0.4, color=color, label=peaks_data['names'][idx])
                
                sum_be = [p['be'] for p in sum_peaks]
                sum_counts = scale_arr([p['counts'] for p in sum_peaks])
                self.axes.plot(sum_be, sum_counts, '--', color='k', linewidth=2.5, label='Total Fit')
        else:
            counts = scale_arr([d['counts'] for d in sorted_data])
            self.axes.plot(be_values, counts, '-', color=COLORS['raw'], linewidth=2.5, label='Raw Data')
            
            if background_data:
                bg_map = {d['be']: d['counts'] for d in background_data}
                bg_counts_aligned = [bg_map.get(be, 0) for be in be_values]
                bg_plot_be = []
                bg_plot_counts = []
                for be, cnt in zip(be_values, bg_counts_aligned):
                    if cnt > 0:
                        bg_plot_be.append(be)
                        bg_plot_counts.append(cnt / scale_factor)
                
                if bg_plot_be:
                    self.axes.plot(bg_plot_be, bg_plot_counts, '-', color=COLORS['background'], linewidth=2.5, label='Shirley BG')
        
        font_axis = {'family': 'Arial', 'fontsize': 20, 'fontweight': 'bold'}
        font_title = {'family': 'Arial', 'fontsize': 24, 'fontweight': 'bold'}
        self.axes.set_xlabel('Binding Energy (eV)', **font_axis)
        self.axes.set_ylabel(ylabel_text, **font_axis)
        self.axes.set_title(region_name, **font_title, pad=15)
        
        for label in (self.axes.get_xticklabels() + self.axes.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(20)
            
        self.axes.grid(True, alpha=0.3)
        import matplotlib.font_manager as fm
        legend_font = fm.FontProperties(family='Arial', size=24)
        self.axes.legend(loc='upper right', prop=legend_font, frameon=True, framealpha=0.9)
        self.axes.set_xlim(max(be_values), min(be_values)) 
        self.fig.tight_layout()
        self.draw()

class PeakWidget(QWidget):
    def __init__(self, peak_number, color, parent=None, callback=None):
        super().__init__(parent)
        self.peak_number = peak_number
        self.color = color
        self.callback = callback
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout()
        self.setStyleSheet(f"border-left: 5px solid {self.color}; background-color: #f9f9f9; border-radius: 4px; margin-bottom: 5px;")
        
        self.name_input = QLineEdit(f"Peak {self.peak_number}")
        self.name_input.editingFinished.connect(self.trigger_save)
        layout.addWidget(QLabel("Name:"), 0, 0)
        layout.addWidget(self.name_input, 0, 1, 1, 3)
        
        self.position_input = QDoubleSpinBox()
        self.position_input.setRange(0, 99999999)
        self.position_input.setDecimals(2)
        self.position_input.setSingleStep(0.1)
        self.position_input.editingFinished.connect(self.trigger_save)
        layout.addWidget(QLabel("Pos (eV):"), 1, 0)
        layout.addWidget(self.position_input, 1, 1)
        
        self.area_input = QDoubleSpinBox()
        self.area_input.setRange(0, 99999999)
        self.area_input.setDecimals(1)
        self.area_input.setSingleStep(100)
        self.area_input.editingFinished.connect(self.trigger_save)
        layout.addWidget(QLabel("Area:"), 1, 2)
        layout.addWidget(self.area_input, 1, 3)
        
        self.fwhm_input = QDoubleSpinBox()
        self.fwhm_input.setRange(0.1, 99999999)
        self.fwhm_input.setDecimals(2)
        self.fwhm_input.setSingleStep(0.1)
        self.fwhm_input.setValue(1.5)
        self.fwhm_input.editingFinished.connect(self.trigger_save)
        layout.addWidget(QLabel("FWHM:"), 2, 0)
        layout.addWidget(self.fwhm_input, 2, 1)
        
        self.lineshape_combo = QComboBox()
        self.lineshape_combo.setEditable(True)
        self.lineshape_combo.addItems(LINE_SHAPES)
        self.lineshape_combo.setCurrentText('GL(30)')
        self.lineshape_combo.currentIndexChanged.connect(self.trigger_save)
        self.lineshape_combo.lineEdit().editingFinished.connect(self.trigger_save)
        layout.addWidget(QLabel("Shape:"), 2, 2)
        layout.addWidget(self.lineshape_combo, 2, 3)
        
        self.remove_btn = QPushButton("Remove Peak")
        self.remove_btn.setStyleSheet("background-color: #ffcccc; color: #cc0000; border: none; padding: 4px;")
        layout.addWidget(self.remove_btn, 3, 0, 1, 4)
        
        self.setLayout(layout)
    
    def trigger_save(self):
        if self.callback: self.callback()

    def get_parameters(self):
        return {
            'name': self.name_input.text(),
            'position': self.position_input.value(),
            'area': self.area_input.value(),
            'fwhm': self.fwhm_input.value(),
            'lineShape': self.lineshape_combo.currentText()
        }
        
    def set_parameters(self, params):
        self.name_input.setText(params['name'])
        self.position_input.setValue(params['position'])
        self.area_input.setValue(params['area'])
        self.fwhm_input.setValue(params['fwhm'])
        self.lineshape_combo.setCurrentText(params['lineShape'])

class XPSAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.regions = []
        self.region_data = {}
        self.current_region = None
        self.peak_widgets = []
        self.undo_manager = UndoManager(limit=15)
        self.init_ui()
        
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_action)
        
    def init_ui(self):
        self.setWindowTitle('XPS Analysis Tool - FFT Convolution Edition')
        self.resize(1300, 850)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        left_layout = QVBoxLayout()
        top_controls = QHBoxLayout()
        load_btn = QPushButton('Load .XY File')
        load_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        load_btn.clicked.connect(self.load_file)
        top_controls.addWidget(load_btn)
        
        self.region_combo = QComboBox()
        self.region_combo.currentTextChanged.connect(self.change_region)
        top_controls.addWidget(self.region_combo)
        top_controls.addStretch()
        left_layout.addLayout(top_controls)
        
        self.canvas = XPSPlotCanvas(self, width=10, height=8)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)
        
        main_layout.addLayout(left_layout, 7)
        
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        bg_group = QGroupBox("1. Background Subtraction")
        bg_layout = QVBoxLayout()
        range_layout = QGridLayout()
        
        range_layout.addWidget(QLabel("Start BE (eV):"), 0, 0)
        self.bg_start_input = QDoubleSpinBox()
        self.bg_start_input.setRange(0, 99999999)
        self.bg_start_input.setDecimals(2)
        range_layout.addWidget(self.bg_start_input, 0, 1)
        
        range_layout.addWidget(QLabel("End BE (eV):"), 1, 0)
        self.bg_end_input = QDoubleSpinBox()
        self.bg_end_input.setRange(0, 99999999)
        self.bg_end_input.setDecimals(2)
        range_layout.addWidget(self.bg_end_input, 1, 1)
        bg_layout.addLayout(range_layout)
        
        self.bg_toggle_btn = QPushButton("Calculate Background")
        self.bg_toggle_btn.setStyleSheet("background-color: #029E73; color: white; padding: 10px; font-weight: bold;")
        self.bg_toggle_btn.clicked.connect(self.toggle_background)
        bg_layout.addWidget(self.bg_toggle_btn)
        
        bg_group.setLayout(bg_layout)
        right_layout.addWidget(bg_group)
        
        self.peak_group = QGroupBox("2. Peak Fitting")
        self.peak_group.setEnabled(False) 
        peak_layout = QVBoxLayout()
        
        add_peak_btn = QPushButton("+ Add Peak")
        add_peak_btn.clicked.connect(self.add_peak_action) 
        peak_layout.addWidget(add_peak_btn)
        
        self.peaks_container_layout = QVBoxLayout()
        peak_layout.addLayout(self.peaks_container_layout)
        
        self.show_peaks_btn = QPushButton("Preview Fit")
        self.show_peaks_btn.setCheckable(True)
        self.show_peaks_btn.clicked.connect(self.toggle_show_peaks)
        peak_layout.addWidget(self.show_peaks_btn)
        
        self.peak_group.setLayout(peak_layout)
        right_layout.addWidget(self.peak_group)
        
        export_btn = QPushButton("Export Results (CSV)")
        export_btn.clicked.connect(self.export_data)
        right_layout.addWidget(export_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(sep)

        comp_btn = QPushButton("\u2697  Composition Analysis\u2026")
        comp_btn.setStyleSheet(
            "background-color: #0173B2; color: white; padding: 12px; "
            "font-weight: bold; font-size: 12pt; border-radius: 4px;"
        )
        comp_btn.clicked.connect(self.open_composition_window)
        right_layout.addWidget(comp_btn)

        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll, 3)

    def save_state(self):
        if not self.current_region: return
        peaks_state = [w.get_parameters() for w in self.peak_widgets]
        r_data = self.region_data[self.current_region]
        bg_state = {
            'isSubtracted': r_data['backgroundSubtracted'],
            'startBE': self.bg_start_input.value(),
            'endBE': self.bg_end_input.value(),
            'backgroundData': copy.deepcopy(r_data.get('backgroundData')),
            'regionDataSubtracted': copy.deepcopy(r_data.get('original', {}).get('data'))
        }
        state = {'peaks': peaks_state, 'bg': bg_state, 'region': self.current_region}
        self.undo_manager.push_state(state)

    def undo_action(self):
        state = self.undo_manager.pop_state()
        if not state: return
        if self.current_region != state['region']:
            self.region_combo.setCurrentText(state['region'])
            
        r_data = self.region_data[self.current_region]
        self.bg_start_input.setValue(state['bg']['startBE'])
        self.bg_end_input.setValue(state['bg']['endBE'])
        r_data['backgroundSubtracted'] = state['bg']['isSubtracted']
        r_data['backgroundData'] = state['bg']['backgroundData']
        if state['bg']['regionDataSubtracted']:
            r_data['original']['data'] = state['bg']['regionDataSubtracted']
            
        if r_data['backgroundSubtracted']:
            self.bg_toggle_btn.setText("Reset / Remove Background")
            self.bg_toggle_btn.setStyleSheet("background-color: #d32f2f; color: white; padding: 10px;")
            self.peak_group.setEnabled(True)
        else:
            self.bg_toggle_btn.setText("Calculate Background")
            self.bg_toggle_btn.setStyleSheet("background-color: #029E73; color: white; padding: 10px; font-weight: bold;")
            self.peak_group.setEnabled(False)

        for w in self.peak_widgets: w.deleteLater()
        self.peak_widgets.clear()
        
        for p_params in state['peaks']:
            self.create_peak_widget_from_params(p_params)
        self.update_plot()

    def add_peak_action(self):
        self.save_state()
        self.add_peak()

    def remove_peak_action(self, widget):
        self.save_state()
        self.remove_peak(widget)

    def toggle_background(self):
        self.save_state()
        self.execute_toggle_background()

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open XY File", "", "XY Files (*.xy);;All Files (*)")
        if filename:
            try:
                self.parse_xy_file(filename)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def parse_xy_file(self, filename):
        self.regions = []
        current_region = None
        metadata = {}
        data = []
        in_data = False
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# Region:'):
                    if current_region and data:
                        self.regions.append({'name': current_region, 'metadata': metadata.copy(), 'data': list(data)})
                    current_region = line.split(':', 1)[1].strip()
                    metadata = {'region': current_region}
                    data = []
                    in_data = False
                elif line.startswith('# Excitation Energy:'):
                    metadata['excitationEnergy'] = float(line.split(':')[1].strip())
                elif line.startswith('# ColumnLabels:'):
                    in_data = True
                elif in_data and line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            ke = float(parts[0])
                            counts = float(parts[1])
                            be = metadata.get('excitationEnergy', 1486.6) - ke
                            data.append({'ke': ke, 'be': be, 'counts': counts})
                        except: continue

        if current_region and data:
            self.regions.append({'name': current_region, 'metadata': metadata, 'data': data})
        
        self.region_data = {}
        for r in self.regions:
            self.region_data[r['name']] = {
                'original': r, 'backgroundSubtracted': False, 
                'backgroundRange': None, 'backgroundData': None, 
                'peaks': [], 'showPeaks': False
            }
        
        self.region_combo.clear()
        self.region_combo.addItems([r['name'] for r in self.regions])
        
        if self.regions:
            self.current_region = self.regions[0]['name']
            self.update_plot()
            self.auto_set_bounds()

    def auto_set_bounds(self):
        if not self.current_region: return
        data = self.region_data[self.current_region]['original']['data']
        be_vals = [d['be'] for d in data]
        if be_vals:
            self.bg_start_input.setValue(max(be_vals)) 
            self.bg_end_input.setValue(min(be_vals))

    def calculate_shirley_background(self, data, start_be, end_be):
        sorted_data = sorted(data, key=lambda x: x['be'], reverse=True)
        start_idx = next((i for i, d in enumerate(sorted_data) if d['be'] <= start_be), 0)
        end_idx = next((i for i, d in enumerate(sorted_data) if d['be'] <= end_be), len(sorted_data)-1)
        if start_idx >= end_idx: return None

        subset = sorted_data[start_idx : end_idx + 1]
        n = len(subset)
        avg_width = 2
        if n > avg_width * 2:
            I_left = np.mean([d['counts'] for d in subset[:avg_width]])
            I_right = np.mean([d['counts'] for d in subset[-avg_width:]])
        else:
            I_left = subset[0]['counts']
            I_right = subset[-1]['counts']

        bg = np.zeros(n)
        bg[:] = I_right
        counts = np.array([d['counts'] for d in subset])
        max_iters = 15
        diff = I_left - I_right
        
        for _ in range(max_iters):
            signal_above_bg = counts - bg
            signal_above_bg[signal_above_bg < 0] = 0
            cumulative_area = np.cumsum(signal_above_bg[::-1])[::-1]
            total_area = cumulative_area[0]
            if total_area == 0: break
            
            new_bg = I_right + diff * (cumulative_area / total_area)
            if np.max(np.abs(new_bg - bg)) < 1e-5:
                bg = new_bg
                break
            bg = new_bg
            
        full_bg = []
        for i, d in enumerate(sorted_data):
            if i < start_idx or i > end_idx:
                full_bg.append({'be': d['be'], 'counts': 0}) 
            else:
                full_bg.append({'be': d['be'], 'counts': bg[i - start_idx]})
        return full_bg

    def execute_toggle_background(self):
        if not self.current_region: return
        r_data = self.region_data[self.current_region]
        
        if r_data['backgroundSubtracted']:
            r_data['backgroundSubtracted'] = False
            self.bg_toggle_btn.setText("Calculate Background")
            self.bg_toggle_btn.setStyleSheet("background-color: #029E73; color: white; padding: 10px; font-weight: bold;")
            self.peak_group.setEnabled(False)
            self.update_plot()
        else:
            v1 = self.bg_start_input.value()
            v2 = self.bg_end_input.value()
            start_be, end_be = max(v1, v2), min(v1, v2)
            bg = self.calculate_shirley_background(r_data['original']['data'], start_be, end_be)
            
            if bg:
                r_data['backgroundSubtracted'] = True
                r_data['backgroundData'] = bg
                data_copy = [d.copy() for d in r_data['original']['data']]
                bg_map = {b['be']: b['counts'] for b in bg}
                
                for d in data_copy:
                    if d['be'] > start_be or d['be'] < end_be:
                        d['countsSubtracted'] = 0
                    else:
                        d['countsSubtracted'] = max(0, d['counts'] - bg_map.get(d['be'], 0))
                
                self.region_data[self.current_region]['original']['data'] = data_copy
                self.bg_toggle_btn.setText("Reset / Remove Background")
                self.bg_toggle_btn.setStyleSheet("background-color: #d32f2f; color: white; padding: 10px;")
                self.peak_group.setEnabled(True)
                self.update_plot()

    def add_peak(self):
        if len(self.peak_widgets) >= 15: return
        color = COLORS['peaks'][len(self.peak_widgets) % len(COLORS['peaks'])]
        widget = PeakWidget(len(self.peak_widgets)+1, color, callback=self.save_state)
        widget.remove_btn.clicked.connect(lambda: self.remove_peak_action(widget))
        self.peaks_container_layout.addWidget(widget)
        self.peak_widgets.append(widget)
        
        center = (self.bg_start_input.value() + self.bg_end_input.value()) / 2
        widget.position_input.setValue(center)

    def create_peak_widget_from_params(self, params):
        color = COLORS['peaks'][len(self.peak_widgets) % len(COLORS['peaks'])]
        widget = PeakWidget(len(self.peak_widgets)+1, color, callback=self.save_state)
        widget.remove_btn.clicked.connect(lambda: self.remove_peak_action(widget))
        widget.set_parameters(params)
        self.peaks_container_layout.addWidget(widget)
        self.peak_widgets.append(widget)

    def remove_peak(self, widget):
        self.peak_widgets.remove(widget)
        widget.deleteLater()
        self.update_plot()

    def calculate_peak_shape(self, be_arr, pos, fwhm, area, shape_str):
        """
        Calculates CasaXPS-style lineshapes in a highly vectorized format.
        LF implements a strict numerical FFT Convolution of an asymmetric Lorentzian 
        and a Gaussian broadening kernel, strictly preserving Area and FWHM scaling.
        """
        x_in = np.atleast_1d(be_arr)
        E = pos
        F = fwhm if fwhm > 0 else 0.001
        
        is_sgl = 'SGL' in shape_str
        is_lf = 'LF' in shape_str
        
        if is_lf:
            alpha, beta, w, m = 1.0, 1.0, 0.0, 30.0
            match = re.search(r'LF\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)(?:,\s*([\d\.]+))?\)', shape_str)
            if match:
                alpha = float(match.group(1))
                beta  = float(match.group(2))
                w     = float(match.group(3))
                if match.group(4) is not None:
                    m = float(match.group(4))
            else:
                match = re.search(r'LF\(([\d\.]+),\s*([\d\.]+)\)', shape_str)
                if match:
                    alpha = float(match.group(1))
                    beta  = float(match.group(2))

            # Dimensionless high-res convolution grid
            u_max = 20.0
            n_pts = 4096
            u = np.linspace(-u_max, u_max, n_pts)
            du = u[1] - u[0]
            
            # Asymmetric Lorentzian (base FWHM = 1 in u-space)
            L = np.where(u > 0, (1 + 4*u**2)**(-alpha), (1 + 4*u**2)**(-beta))
            
            # Gaussian Broadening Kernel
            G_FWHM = max(w / 100.0, 0.001)
            ln2 = np.log(2)
            G = np.exp(-4 * ln2 * (u)**2 / (G_FWHM**2))
            G /= np.sum(G) * du
            
            # True FFT Convolution
            C = fftconvolve(L, G, mode='same') * du
            
            # Extract FWHM of the numerical convolution
            max_idx = np.argmax(C)
            max_val = C[max_idx]
            half_val = max_val / 2.0
            
            left_idx = np.argmin(np.abs(C[:max_idx] - half_val))
            right_idx = max_idx + np.argmin(np.abs(C[max_idx:] - half_val))
            conv_fwhm = u[right_idx] - u[left_idx]
            if conv_fwhm <= 0: conv_fwhm = 1.0
            
            # Interpolate convolution back to exact user-requested FWHM
            u_target = (x_in - E) * (conv_fwhm / F)
            y_out = np.interp(u_target, u, C, left=0, right=0)
            
        else:
            # Analytic Symmetric Peaks (GL / SGL)
            m = 30.0
            match = re.search(r'\(([\d\.]+)\)', shape_str)
            if match: m = float(match.group(1))
                
            dx = x_in - E
            ln2 = np.log(2)
            g = np.exp(-4 * ln2 * (dx/F)**2)
            l = 1 / (1 + 4 * (dx/F)**2)
            mix = m / 100.0
            
            if is_sgl: y_out = (1 - mix)*g + mix*l
            else:      y_out = (g**(1-mix)) * (l**mix)
                
        # Strict Area Normalization (Essential for Composition Matrix)
        u_dense = np.linspace(E - 10*F, E + 10*F, 2048)
        
        if is_lf:
            u_t = (u_dense - E) * (conv_fwhm / F)
            C_dense = np.interp(u_t, u, C, left=0, right=0)
        else:
            dx_d = u_dense - E
            g_d = np.exp(-4 * np.log(2) * (dx_d/F)**2)
            l_d = 1 / (1 + 4 * (dx_d/F)**2)
            if is_sgl: C_dense = (1 - mix)*g_d + mix*l_d
            else:      C_dense = (g_d**(1-mix)) * (l_d**mix)
            
        true_area = np.trapz(C_dense, u_dense)
        if true_area > 0:
            y_out = y_out * (area / true_area)
            
        return y_out[0] if np.isscalar(be_arr) else y_out

    def toggle_show_peaks(self):
        self.update_plot()

    def update_plot(self):
        if not self.current_region: return
        r = self.region_data[self.current_region]
        
        peaks_info = None
        if self.show_peaks_btn.isChecked() and self.peak_widgets:
            peaks_res = []
            names = []
            data = r['original']['data']
            be_arr = np.array([d['be'] for d in data])
            
            for w in self.peak_widgets:
                p = w.get_parameters()
                # Vectorized Evaluation
                y_arr = self.calculate_peak_shape(be_arr, p['position'], p['fwhm'], p['area'], p['lineShape'])
                y_vals = [{'be': be, 'counts': y} for be, y in zip(be_arr, y_arr)]
                peaks_res.append(y_vals)
                names.append(p['name'])
            
            sum_res = []
            for i in range(len(data)):
                total = sum(peaks_res[j][i]['counts'] for j in range(len(peaks_res)))
                sum_res.append({'be': data[i]['be'], 'counts': total})
            
            peaks_info = {'peaks': peaks_res, 'sum': sum_res, 'names': names}
            
        self.canvas.plot_spectrum(
            r['original']['data'], 
            r['original']['metadata']['region'],
            background_subtracted=r['backgroundSubtracted'],
            background_data=r.get('backgroundData'),
            peaks_data=peaks_info,
            show_peaks=self.show_peaks_btn.isChecked()
        )

    def change_region(self, name):
        if name:
            self.current_region = name
            self.update_plot()
            self.auto_set_bounds()
            
    def open_composition_window(self):
        if not self.region_data:
            QMessageBox.warning(self, "No Data", "Please load an .XY file first.")
            return
        dlg = CompositionWindow(self.region_data, self.calculate_peak_shape, self.calculate_shirley_background, parent=self)
        dlg.show()

    def export_data(self):
        if not self.current_region: return
        r_data = self.region_data[self.current_region]
        data = r_data['original']['data']
        sorted_data = sorted(data, key=lambda x: x['ke'])
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save Analysis Results", f"{self.current_region}_fit.csv", "CSV Files (*.csv)")
        if not filename: return
            
        try:
            with open(filename, 'w') as f:
                f.write(f"# Region: {self.current_region}\n")
                peaks_res = []
                peak_names = []
                be_arr = np.array([d['be'] for d in sorted_data])
                
                if self.peak_widgets:
                    for w in self.peak_widgets:
                        p = w.get_parameters()
                        peak_names.append(p['name'])
                        # Vectorized Evaluation
                        y_vals = self.calculate_peak_shape(be_arr, p['position'], p['fwhm'], p['area'], p['lineShape']).tolist()
                        peaks_res.append(y_vals)

                headers = ["Kinetic Energy (eV)", "Binding Energy (eV)", "Raw Counts", "Background", "Counts After Bg Sub"]
                if peaks_res:
                    headers.append("Total Envelope")
                    for name in peak_names:
                        headers.append(f"{name} (Fit)")
                f.write(",".join(headers) + "\n")
                
                bg_data = r_data.get('backgroundData')
                bg_map = {d['be']: d['counts'] for d in bg_data} if bg_data else {}
                start_be = max(self.bg_start_input.value(), self.bg_end_input.value())
                end_be = min(self.bg_start_input.value(), self.bg_end_input.value())
                
                for i, d in enumerate(sorted_data):
                    be = d['be']
                    ke = d['ke']
                    raw = d['counts']
                    bg_val = bg_map.get(be, 0)
                    sub_val = 0 if (be > start_be or be < end_be) else max(0, raw - bg_val)
                    
                    row = [f"{ke:.2f}", f"{be:.2f}", f"{raw:.1f}", f"{bg_val:.1f}", f"{sub_val:.1f}"]
                    
                    if peaks_res:
                        total_fit = sum(peaks_res[p_idx][i] for p_idx in range(len(peaks_res)))
                        row.append(f"{total_fit:.1f}")
                        for p_idx in range(len(peaks_res)):
                            row.append(f"{peaks_res[p_idx][i]:.1f}")
                    f.write(",".join(row) + "\n")
            QMessageBox.information(self, "Success", f"Exported: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

DEFAULT_ASF = {
    'C':  0.296, 'N':  0.477, 'O':  0.711, 'Si': 0.339, 
    'Ta': 3.082, 'Hf': 2.639, 'Ti': 2.001, 'W':  3.523, 'Al': 0.234,
}

REGION_ELEMENT_MAP = {
    'c1s':  'C',  'c 1s': 'C', 'n1s':  'N',  'n 1s': 'N',
    'o1s':  'O',  'o 1s': 'O', 'si2p': 'Si', 'si 2p': 'Si',
    'ta4f': 'Ta', 'ta 4f': 'Ta', 'hf4f': 'Hf', 'hf 4f': 'Hf',
    'ti2p': 'Ti', 'ti 2p': 'Ti', 'w4f':  'W',  'w 4f':  'W',
    'al2p': 'Al', 'al 2p': 'Al',
}

def guess_element(region_name):
    name_lower = region_name.lower().replace('_', ' ')
    for key, elem in REGION_ELEMENT_MAP.items():
        if key in name_lower: return elem
    return None

class ExcludePeakWidget(QWidget):
    changed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        self.name_edit = QLineEdit("Ta 4p"); self.name_edit.setFixedWidth(80)
        self.pos_spin  = QDoubleSpinBox(); self.pos_spin.setRange(0, 9999); self.pos_spin.setDecimals(2); self.pos_spin.setValue(400.5); self.pos_spin.setFixedWidth(80)
        self.fwhm_spin = QDoubleSpinBox(); self.fwhm_spin.setRange(0.1, 99); self.fwhm_spin.setDecimals(2); self.fwhm_spin.setValue(2.0);  self.fwhm_spin.setFixedWidth(70)
        self.shape_cb  = QComboBox(); self.shape_cb.addItems(LINE_SHAPES); self.shape_cb.setFixedWidth(90)
        self.area_label = QLabel("0.00"); self.area_label.setFixedWidth(80); self.area_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.remove_btn = QPushButton("✕"); self.remove_btn.setFixedWidth(28); self.remove_btn.setStyleSheet("color:#c00; font-weight:bold; border:none;")

        for w, lbl in [(self.name_edit, "Name"), (self.pos_spin, "Pos(eV)"),
                       (self.fwhm_spin, "FWHM"), (self.shape_cb, "Shape")]:
            lay.addWidget(QLabel(lbl)); lay.addWidget(w)
        lay.addWidget(QLabel("Area:")); lay.addWidget(self.area_label); lay.addWidget(self.remove_btn); lay.addStretch()

        self.pos_spin.valueChanged.connect(self.changed)
        self.fwhm_spin.valueChanged.connect(self.changed)
        self.shape_cb.currentIndexChanged.connect(self.changed)

    def get_params(self):
        return {'name': self.name_edit.text(), 'position': self.pos_spin.value(),
                'fwhm': self.fwhm_spin.value(), 'lineShape': self.shape_cb.currentText()}

    def set_area_display(self, area):
        self.area_label.setText(f"{area:.2f}")

class RegionCompositionWidget(QGroupBox):
    recalculate_requested = pyqtSignal()
    def __init__(self, region_name, region_data, calc_peak_shape_fn, parent=None):
        super().__init__(region_name, parent)
        self.region_name      = region_name
        self.region_data      = region_data
        self.calc_peak_shape  = calc_peak_shape_fn
        self.exclude_widgets  = []
        self._build_ui()
        self._detect_element()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Element:"))
        self.elem_edit = QLineEdit(); self.elem_edit.setFixedWidth(50)
        self.elem_edit.textChanged.connect(self.recalculate_requested)
        row1.addWidget(self.elem_edit)

        row1.addWidget(QLabel("ASF:"))
        self.asf_spin = QDoubleSpinBox()
        self.asf_spin.setRange(0.001, 99); self.asf_spin.setDecimals(4)
        self.asf_spin.setSingleStep(0.01); self.asf_spin.setFixedWidth(80)
        self.asf_spin.valueChanged.connect(self.recalculate_requested)
        row1.addWidget(self.asf_spin)

        row1.addWidget(QLabel("Include:"))
        self.include_cb = QCheckBox(); self.include_cb.setChecked(True)
        self.include_cb.stateChanged.connect(self.recalculate_requested)
        row1.addWidget(self.include_cb); row1.addStretch(); lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("BG range  Start:"))
        self.start_spin = QDoubleSpinBox(); self.start_spin.setRange(0, 9999); self.start_spin.setDecimals(2); self.start_spin.setFixedWidth(80); row2.addWidget(self.start_spin)
        row2.addWidget(QLabel("End:"))
        self.end_spin   = QDoubleSpinBox(); self.end_spin.setRange(0, 9999);   self.end_spin.setDecimals(2); self.end_spin.setFixedWidth(80); row2.addWidget(self.end_spin)
        apply_btn = QPushButton("Recalc BG"); apply_btn.setFixedWidth(90)
        apply_btn.clicked.connect(self._recalc_bg_and_emit)
        row2.addWidget(apply_btn); row2.addStretch(); lay.addLayout(row2)

        excl_header = QHBoxLayout()
        excl_header.addWidget(QLabel("<b>Exclude / subtract peaks:</b>"))
        add_excl_btn = QPushButton("+ Add Peak"); add_excl_btn.setFixedWidth(90)
        add_excl_btn.clicked.connect(self._add_exclude_peak)
        excl_header.addWidget(add_excl_btn); excl_header.addStretch(); lay.addLayout(excl_header)
        self.excl_container = QVBoxLayout(); lay.addLayout(self.excl_container)

        row3 = QHBoxLayout()
        self.raw_area_label  = QLabel("Raw area: —")
        self.excl_area_label = QLabel("Excluded: —")
        self.net_area_label  = QLabel("<b>Net area: —</b>")
        self.atpct_label     = QLabel("<b>at%: —</b>")
        self.atpct_label.setStyleSheet("color:#0173B2; font-size:14pt;")
        for w in (self.raw_area_label, self.excl_area_label, self.net_area_label, self.atpct_label): row3.addWidget(w)
        row3.addStretch(); lay.addLayout(row3)

    def _detect_element(self):
        elem = guess_element(self.region_name)
        if elem:
            self.elem_edit.setText(elem)
            self.asf_spin.setValue(DEFAULT_ASF.get(elem, 1.0))
        else:
            self.elem_edit.setText("?")
            self.asf_spin.setValue(1.0)

        data = self.region_data['original']['data']
        be_vals = [d['be'] for d in data]
        if be_vals:
            self.start_spin.setValue(max(be_vals))
            self.end_spin.setValue(min(be_vals))
        if self.region_data.get('backgroundData'):
            bg = self.region_data['backgroundData']
            active = [d for d in bg if d['counts'] > 0]
            if active:
                active_bes = [d['be'] for d in active]
                self.start_spin.setValue(max(active_bes))
                self.end_spin.setValue(min(active_bes))

    def _add_exclude_peak(self):
        w = ExcludePeakWidget()
        w.changed.connect(self._on_exclude_changed)
        w.remove_btn.clicked.connect(lambda: self._remove_exclude(w))
        self.excl_container.addWidget(w)
        self.exclude_widgets.append(w)
        self.recalculate_requested.emit()

    def _remove_exclude(self, w):
        self.exclude_widgets = [x for x in self.exclude_widgets if x is not w]
        w.deleteLater()
        self.recalculate_requested.emit()

    def _on_exclude_changed(self): self.recalculate_requested.emit()

    def _recalc_bg_and_emit(self):
        self._recompute_bg()
        self.recalculate_requested.emit()

    def _recompute_bg(self):
        from copy import deepcopy
        data = self.region_data['original']['data']
        start_be, end_be = max(self.start_spin.value(), self.end_spin.value()), min(self.start_spin.value(), self.end_spin.value())
        parent = self.parent()
        while parent and not hasattr(parent, 'calculate_shirley_background'): parent = parent.parent()
        if parent is None: return
        bg = parent.calculate_shirley_background(data, start_be, end_be)
        if bg is None: return

        self.region_data['backgroundSubtracted'] = True
        self.region_data['backgroundData'] = bg
        bg_map = {b['be']: b['counts'] for b in bg}
        data_copy = deepcopy(data)
        for d in data_copy:
            d['countsSubtracted'] = 0 if (d['be'] > start_be or d['be'] < end_be) else max(0, d['counts'] - bg_map.get(d['be'], 0))
        self.region_data['original']['data'] = data_copy

    def compute_net_area(self):
        included, elem, asf = self.include_cb.isChecked(), self.elem_edit.text().strip(), self.asf_spin.value()
        data = self.region_data['original']['data']
        start_be, end_be = max(self.start_spin.value(), self.end_spin.value()), min(self.start_spin.value(), self.end_spin.value())

        if self.region_data.get('backgroundSubtracted'):
            pts = [(d['be'], d.get('countsSubtracted', 0)) for d in data if end_be <= d['be'] <= start_be]
        else:
            pts = [(d['be'], d['counts']) for d in data if end_be <= d['be'] <= start_be]

        if len(pts) < 2: return 0.0, 0.0, 0.0, asf, elem, included

        pts.sort(key=lambda x: x[0])
        be_arr, cnt_arr = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        raw_area = float(np.trapz(cnt_arr, be_arr))

        excl_area = 0.0
        for ew in self.exclude_widgets:
            p = ew.get_params()
            peak_vals = self.calc_peak_shape(be_arr, p['position'], p['fwhm'], 1.0, p['lineShape'])
            norm = float(np.trapz(peak_vals, be_arr))
            if norm > 0:
                sigma = p['fwhm'] / (2 * np.sqrt(2 * np.log(2)))
                mask = np.abs(be_arr - p['position']) <= 3 * sigma
                if mask.sum() >= 2:
                    scale = np.dot(cnt_arr[mask], peak_vals[mask]) / np.dot(peak_vals[mask], peak_vals[mask])
                    scale = max(scale, 0)
                    excl_peak_area = float(np.trapz(scale * peak_vals[mask], be_arr[mask]))
                    ew.set_area_display(excl_peak_area)
                    excl_area += excl_peak_area
                else: ew.set_area_display(0.0)

        net_area = max(raw_area - abs(excl_area), 0.0)
        return raw_area, excl_area, net_area, asf, elem, included

class CompositionWindow(QDialog):
    def __init__(self, region_data, calc_peak_shape_fn, shirley_fn, parent=None):
        super().__init__(parent)
        self.region_data       = region_data
        self.calc_peak_shape   = calc_peak_shape_fn
        self.shirley_fn        = shirley_fn
        self.region_widgets    = {}

        self.setWindowTitle("XPS Composition Analysis")
        self.resize(1050, 780)
        self._build_ui()
        self._populate_regions()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        title = QLabel("Atomic Composition Calculator")
        title.setStyleSheet("font-size:18pt; font-weight:bold; color:#0173B2; padding:8px;")
        outer.addWidget(title)

        desc = QLabel("For each region: set the element and ASF, optionally add 'exclude peaks' "
                      "(e.g. Ta 4p inside N 1s), then click <b>Recalculate</b>. "
                      "Uncheck <i>Include</i> to omit a region from the at% sum.")
        desc.setWordWrap(True)
        outer.addWidget(desc)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._cards_widget  = QWidget()
        self._cards_layout  = QVBoxLayout(self._cards_widget)
        self._cards_layout.addStretch()
        left_scroll.setWidget(self._cards_widget)
        splitter.addWidget(left_scroll)

        right_widget = QWidget()
        right_lay    = QVBoxLayout(right_widget)
        right_lay.addWidget(QLabel("<b>Composition Results</b>"))

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Region", "Element", "Net Area", "ASF", "at%"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        right_lay.addWidget(self.results_table, stretch=1)

        self.bar_canvas = XPSPlotCanvas(self, width=5, height=4, dpi=80)
        right_lay.addWidget(self.bar_canvas, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([580, 470])

        btn_row = QHBoxLayout()
        recalc_btn = QPushButton("⟳  Recalculate All")
        recalc_btn.setStyleSheet("background:#029E73; color:white; font-weight:bold; padding:10px; font-size:12pt;")
        recalc_btn.clicked.connect(self.recalculate_all)

        export_btn = QPushButton("Export CSV")
        export_btn.setStyleSheet("background:#0173B2; color:white; padding:10px; font-size:12pt;")
        export_btn.clicked.connect(self.export_csv)

        btn_row.addWidget(recalc_btn); btn_row.addWidget(export_btn); btn_row.addStretch(); outer.addLayout(btn_row)

    def _populate_regions(self):
        for name, data in self.region_data.items():
            w = RegionCompositionWidget(name, data, self.calc_peak_shape, parent=self)
            w.recalculate_requested.connect(self.recalculate_all)
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, w)
            self.region_widgets[name] = w
        self.recalculate_all()

    def recalculate_all(self):
        results = []
        for name, w in self.region_widgets.items():
            raw, excl, net, asf, elem, included = w.compute_net_area()
            w.raw_area_label.setText(f"Raw area: {raw:.1f}")
            w.excl_area_label.setText(f"Excluded: {excl:.1f}")
            w.net_area_label.setText(f"<b>Net area: {net:.1f}</b>")
            if included and asf > 0:
                results.append({'region': name, 'elem': elem, 'net': net, 'asf': asf, 'norm': net / asf})
            else:
                w.atpct_label.setText("<b>at%: excluded</b>")

        total_norm = sum(r['norm'] for r in results)
        self.results_table.setRowCount(0)
        for r in results:
            at_pct = 100.0 * r['norm'] / total_norm if total_norm > 0 else 0.0
            self.region_widgets[r['region']].atpct_label.setText(f"<b>at%: {at_pct:.2f}%</b>")
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            for col, txt in enumerate([r['region'], r['elem'], f"{r['net']:.1f}", f"{r['asf']:.4f}", f"{at_pct:.2f}%"]):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 4:
                    item.setFont(QFont("Arial", 11, QFont.Bold))
                    item.setBackground(QColor(230 - int(min(at_pct / 100.0, 1.0) * 180), 240, 255))
                self.results_table.setItem(row, col, item)

        self._draw_bar_chart(results, total_norm)

    def _draw_bar_chart(self, results, total_norm):
        ax = self.bar_canvas.axes
        ax.clear()
        if not results or total_norm == 0: self.bar_canvas.draw(); return

        labels   = [f"{r['elem']}\\n({r['region']})" for r in results]
        at_pcts  = [100.0 * r['norm'] / total_norm for r in results]
        colors   = [COLORS['peaks'][i % len(COLORS['peaks'])] for i in range(len(results))]

        bars = ax.bar(labels, at_pcts, color=colors, edgecolor='k', linewidth=0.6)
        for bar, val in zip(bars, at_pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{val:.1f}%", ha='center', va='bottom', fontname='Arial', fontsize=9, fontweight='bold')

        ax.set_ylabel("Atomic %", fontname='Arial', fontsize=11, fontweight='bold')
        ax.set_title("Elemental Composition", fontname='Arial', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(at_pcts) * 1.18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(9)
        ax.grid(axis='y', alpha=0.3)
        self.bar_canvas.fig.tight_layout()
        self.bar_canvas.draw()

    def export_csv(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Composition", "composition.csv", "CSV Files (*.csv)")
        if not filename: return
        try:
            rows = [[self.results_table.item(row, c).text() for c in range(self.results_table.columnCount())] for row in range(self.results_table.rowCount())]
            with open(filename, 'w') as f:
                f.write("Region,Element,Net Area,ASF,at%\\n")
                for row in rows: f.write(",".join(row) + "\\n")
            QMessageBox.information(self, "Exported", f"Saved to:\\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = XPSAnalysisGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()