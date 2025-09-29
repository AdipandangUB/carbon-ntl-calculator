"""
Script Analisis Nighttime Light pada Streamlit
FOKUS PADA VISUALISASI GEOSPASIAL NTL DAN PERHITUNGAN KARBON - FIXED VERSION
"""

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import tempfile
import os
import folium
from streamlit_folium import st_folium
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------
# KONFIGURASI HALAMAN
# ----------------------------

st.set_page_config(
    page_title="NTL Carbon Analysis Tool", 
    layout="wide",
    page_icon="🌃"
)

st.title("🌃 Nighttime Lights Carbon Analysis Tool")
st.markdown("**Analisis Nighttime Lights dan Estimasi Emisi Karbon**")

# ----------------------------
# FUNGSI PERHITUNGAN KARBON - FIXED
# ----------------------------

def calculate_carbon_from_ntl(ntl_data, conversion_method="Elvidge_2009", custom_coef=0.001):
    """
    Menghitung estimasi karbon dari data nighttime lights menggunakan berbagai metode
    """
    # Mask NaN values
    valid_data = ntl_data[~np.isnan(ntl_data)]
    
    if len(valid_data) == 0:
        return 0, {}
    
    # Konversi berdasarkan metode yang dipilih
    if conversion_method == "Elvidge_2009":
        carbon_coef = 0.0005
        carbon_estimate = np.sum(valid_data) * carbon_coef
        
    elif conversion_method == "Shi_2016":
        carbon_coef = 0.0008
        carbon_estimate = np.sum(valid_data) * carbon_coef
        
    elif conversion_method == "Custom_Linear":
        carbon_coef = custom_coef
        carbon_estimate = np.sum(valid_data) * carbon_coef
        
    elif conversion_method == "Power_Law":
        carbon_estimate = np.sum(np.power(valid_data, 1.2)) * 0.0001
        carbon_coef = 0.0001  # untuk statistik
        
    else:
        carbon_coef = 0.0006
        carbon_estimate = np.sum(valid_data) * carbon_coef
    
    # Statistik tambahan
    stats = {
        'total_ntl': np.sum(valid_data),
        'mean_ntl': np.mean(valid_data),
        'max_ntl': np.max(valid_data),
        'pixel_count': len(valid_data),
        'carbon_coefficient': carbon_coef,
        'carbon_per_pixel': carbon_estimate / len(valid_data) if len(valid_data) > 0 else 0
    }
    
    return carbon_estimate, stats

def apply_threshold_filter(ntl_data, threshold_type="auto", custom_threshold=None):
    """
    Apply threshold untuk memfilter area dengan NTL signifikan
    """
    valid_data = ntl_data.copy()
    valid_data[np.isnan(valid_data)] = 0
    
    if threshold_type == "auto":
        threshold = np.mean(valid_data) + 0.5 * np.std(valid_data)
    elif threshold_type == "percentile":
        if np.sum(valid_data > 0) > 0:
            threshold = np.percentile(valid_data[valid_data > 0], 75)
        else:
            threshold = 0
    elif threshold_type == "manual" and custom_threshold is not None:
        threshold = custom_threshold
    else:
        threshold = 0
    
    # Apply threshold
    filtered_data = ntl_data.copy()
    filtered_data[filtered_data < threshold] = 0
    
    return filtered_data

def generate_carbon_report(raster_paths, conversion_method="Elvidge_2009", threshold_type="auto", custom_coef=0.001):
    """
    Generate laporan karbon komprehensif untuk multiple datasets
    """
    carbon_results = []
    
    for i, path in enumerate(raster_paths):
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                data[data == src.nodata] = np.nan
                
                # Apply threshold filter
                filtered_data = apply_threshold_filter(data, threshold_type)
                
                # Calculate carbon estimate
                carbon_estimate, stats = calculate_carbon_from_ntl(filtered_data, conversion_method, custom_coef)
                
                result = {
                    'Dataset': f"NTL {i+1}",
                    'Carbon_Estimate_Ton': carbon_estimate,
                    'Carbon_Estimate_KTon': carbon_estimate / 1000,
                    'Total_NTL': stats['total_ntl'],
                    'Pixel_Count': stats['pixel_count'],
                    'Active_Pixels': np.sum(filtered_data > 0),
                    'Carbon_Per_Pixel': stats['carbon_per_pixel'],
                    'Mean_NTL': stats['mean_ntl'],
                    'Max_NTL': stats['max_ntl']
                }
                carbon_results.append(result)
                
        except Exception as e:
            st.error(f"Error processing file {i+1}: {str(e)}")
            # Tambahkan result kosong untuk konsistensi
            carbon_results.append({
                'Dataset': f"NTL {i+1}",
                'Carbon_Estimate_Ton': 0,
                'Carbon_Estimate_KTon': 0,
                'Total_NTL': 0,
                'Pixel_Count': 0,
                'Active_Pixels': 0,
                'Carbon_Per_Pixel': 0,
                'Mean_NTL': 0,
                'Max_NTL': 0
            })
    
    return pd.DataFrame(carbon_results)

# ----------------------------
# FUNGSI VISUALISASI KARBON - FIXED
# ----------------------------

def create_ntl_colormap():
    """Membuat colormap khusus untuk nighttime lights"""
    colors = ['black', 'darkblue', 'blue', 'cyan', 'yellow', 'white']
    return LinearSegmentedColormap.from_list('ntl_colormap', colors, N=256)

def plot_carbon_distribution(ntl_data, carbon_data, title="Carbon Distribution"):
    """Visualisasi distribusi karbon"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Data NTL asli
    ntl_cmap = create_ntl_colormap()
    im1 = axes[0].imshow(ntl_data, cmap=ntl_cmap)
    axes[0].set_title('Nighttime Lights Original')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Plot 2: Estimasi karbon
    carbon_cmap = plt.cm.YlOrRd
    im2 = axes[1].imshow(carbon_data, cmap=carbon_cmap)
    axes[1].set_title('Carbon Estimate')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_carbon_trends(carbon_df):
    """Plot trend karbon dari waktu ke waktu"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Trend total karbon
    axes[0,0].plot(range(len(carbon_df)), carbon_df['Carbon_Estimate_Ton'], 'o-', linewidth=2, markersize=8)
    axes[0,0].set_title('Trend Total Estimasi Karbon')
    axes[0,0].set_xlabel('Dataset')
    axes[0,0].set_ylabel('Karbon (Ton)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Trend kepadatan karbon
    axes[0,1].plot(range(len(carbon_df)), carbon_df['Carbon_Per_Pixel'], 's-', linewidth=2, markersize=8)
    axes[0,1].set_title('Trend Kepadatan Karbon per Pixel')
    axes[0,1].set_xlabel('Dataset')
    axes[0,1].set_ylabel('Karbon per Pixel (Ton)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Total NTL
    axes[1,0].bar(range(len(carbon_df)), carbon_df['Total_NTL'])
    axes[1,0].set_title('Total Nighttime Lights')
    axes[1,0].set_xlabel('Dataset')
    axes[1,0].set_ylabel('Total NTL')
    
    # Plot 4: Perbandingan NTL vs Karbon
    axes[1,1].scatter(carbon_df['Total_NTL'], carbon_df['Carbon_Estimate_Ton'], s=100, alpha=0.7)
    axes[1,1].set_title('Hubungan NTL vs Estimasi Karbon')
    axes[1,1].set_xlabel('Total NTL')
    axes[1,1].set_ylabel('Estimasi Karbon (Ton)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ----------------------------
# FUNGSI VISUALISASI GEOSPASIAL - FIXED
# ----------------------------

def plot_geospatial_ntl(raster_path, title="Nighttime Lights"):
    """Visualisasi geospasial raster NTL dengan Matplotlib"""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            # Handle no data values
            data[data == src.nodata] = np.nan
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot raster dengan colormap khusus NTL
            ntl_cmap = create_ntl_colormap()
            im = ax.imshow(data, cmap=ntl_cmap, 
                          extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Radiance (nW/cm²/sr)')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
    except Exception as e:
        st.error(f"Error dalam visualisasi raster: {str(e)}")
        return None

def create_interactive_ntl_map(raster_paths, year_labels=None):
    """Membuat peta interaktif dengan Folium untuk data NTL"""
    try:
        if not raster_paths:
            return None
            
        # Gunakan raster pertama untuk center map
        with rasterio.open(raster_paths[0]) as src:
            bounds = src.bounds
            center_lat = (bounds.top + bounds.bottom) / 2
            center_lon = (bounds.left + bounds.right) / 2
        
        # Buat peta dasar
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=8, 
                      tiles='CartoDB dark_matter')
        
        # Tambahkan tile layers alternatif
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Untuk setiap raster, tambahkan sebagai overlay
        for i, raster_path in enumerate(raster_paths):
            try:
                with rasterio.open(raster_path) as src:
                    bounds = src.bounds
                    year_label = year_labels[i] if year_labels and i < len(year_labels) else f"Year {i+1}"
                    
                    # Convert raster to PNG untuk overlay
                    data = src.read(1)
                    data[data == src.nodata] = 0
                    
                    # Normalisasi data untuk visualisasi
                    if np.nanmax(data) > np.nanmin(data):
                        data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
                    else:
                        data_norm = data
                    
                    # Simpan sebagai PNG sementara
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        plt.imsave(tmp_file.name, data_norm, cmap=create_ntl_colormap())
                        
                        # Add raster overlay ke peta
                        img_overlay = folium.raster_layers.ImageOverlay(
                            name=year_label,
                            image=tmp_file.name,
                            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            opacity=0.7,
                            interactive=True,
                            cross_origin=False
                        ).add_to(m)
                        
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
                        
            except Exception as e:
                st.warning(f"Error processing raster {i+1} for map: {str(e)}")
                continue
        
        # Tambahkan layer control
        folium.LayerControl().add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error membuat peta interaktif: {str(e)}")
        return None

def plot_ntl_comparison(raster_paths, titles=None):
    """Plot komparasi multiple NTL raster dalam grid"""
    n_rasters = len(raster_paths)
    if n_rasters == 0:
        return None
        
    n_cols = min(2, n_rasters)  # Reduced to 2 columns for better visibility
    n_rows = (n_rasters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_rasters == 1:
        axes = np.array([axes])
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, (ax, raster_path) in enumerate(zip(axes_flat, raster_paths)):
        try:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                data[data == src.nodata] = np.nan
                bounds = src.bounds
                
                ntl_cmap = create_ntl_colormap()
                im = ax.imshow(data, cmap=ntl_cmap,
                              extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
                
                title = titles[i] if titles and i < len(titles) else f"NTL {i+1}"
                ax.set_title(title, fontsize=12)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar untuk setiap subplot
                plt.colorbar(im, ax=ax, shrink=0.8)
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Error Loading Data")
    
    # Sembunyikan axes yang tidak terpakai
    for j in range(len(raster_paths), len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def generate_ntl_statistics(raster_paths):
    """Generate statistics untuk data NTL"""
    stats_data = []
    
    for i, path in enumerate(raster_paths):
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                data[data == src.nodata] = np.nan
                
                stats = {
                    'Dataset': f"NTL {i+1}",
                    'Min': np.nanmin(data),
                    'Max': np.nanmax(data),
                    'Mean': np.nanmean(data),
                    'Std': np.nanstd(data),
                    'Area (px)': np.sum(~np.isnan(data))
                }
                stats_data.append(stats)
        except Exception as e:
            st.error(f"Error processing file {i+1} for statistics: {str(e)}")
    
    return pd.DataFrame(stats_data)

# ----------------------------
# FUNGSI UTAMA ANALISIS KARBON - FIXED
# ----------------------------

def setup_carbon_analysis():
    """Setup untuk analisis karbon dari NTL"""
    
    st.header("🌿 Estimasi Karbon dari Nighttime Lights")
    st.markdown("Perhitungan estimasi emisi karbon berdasarkan data nighttime lights menggunakan berbagai metode konversi")
    
    # Upload data raster
    raster_files = st.file_uploader(
        "Pilih file TIFF raster NTL untuk analisis karbon", 
        type=["tif", "tiff"], 
        accept_multiple_files=True,
        help="Unggah file raster NTL untuk perhitungan estimasi karbon",
        key="carbon_upload"
    )
    
    if raster_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_paths = []
            for i, uploaded_file in enumerate(raster_files):
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raster_paths.append(file_path)
            
            st.success(f"✅ {len(raster_paths)} file raster berhasil diunggah untuk analisis karbon")
            
            # Konfigurasi analisis karbon
            col1, col2, col3 = st.columns(3)
            
            custom_coef = 0.001  # default value
            
            with col1:
                conversion_method = st.selectbox(
                    "Metode Konversi NTL-Karbon",
                    ["Elvidge_2009", "Shi_2016", "Power_Law", "Custom_Linear"],
                    help="Pilih metode konversi dari NTL ke estimasi karbon"
                )
                
                if conversion_method == "Custom_Linear":
                    custom_coef = st.number_input(
                        "Koefisien Konversi Kustom",
                        min_value=0.0001,
                        max_value=0.01,
                        value=0.001,
                        step=0.0001,
                        help="Koefisien konversi: Carbon = NTL × Koefisien"
                    )
            
            with col2:
                threshold_type = st.selectbox(
                    "Filter Threshold",
                    ["auto", "percentile", "manual", "none"],
                    help="Filter area dengan NTL signifikan untuk perhitungan"
                )
                
                custom_threshold = None
                if threshold_type == "manual":
                    custom_threshold = st.number_input(
                        "Nilai Threshold Manual",
                        min_value=0.0,
                        max_value=100.0,
                        value=5.0,
                        step=0.1
                    )
            
            with col3:
                analysis_type = st.selectbox(
                    "Tipe Analisis",
                    ["Laporan Karbon", "Visualisasi Distribusi", "Analisis Trend", "Perbandingan Metode"]
                )
            
            # Generate carbon report
            try:
                carbon_df = generate_carbon_report(raster_paths, conversion_method, threshold_type, custom_coef)
                
                # Tampilkan hasil berdasarkan tipe analisis
                if analysis_type == "Laporan Karbon":
                    st.subheader("📊 Laporan Estimasi Karbon")
                    
                    # Summary metrics
                    if len(carbon_df) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        total_carbon = carbon_df['Carbon_Estimate_Ton'].sum()
                        
                        with col1:
                            st.metric("Total Estimasi Karbon", f"{total_carbon:,.0f} Ton")
                        with col2:
                            st.metric("Rata-rata per Dataset", f"{carbon_df['Carbon_Estimate_Ton'].mean():,.0f} Ton")
                        with col3:
                            st.metric("Dataset Tertinggi", f"{carbon_df['Carbon_Estimate_Ton'].max():,.0f} Ton")
                        with col4:
                            st.metric("Dataset Terendah", f"{carbon_df['Carbon_Estimate_Ton'].min():,.0f} Ton")
                        
                        # Tabel detail
                        st.dataframe(carbon_df.style.format({
                            'Carbon_Estimate_Ton': '{:,.0f}',
                            'Carbon_Estimate_KTon': '{:,.1f}',
                            'Total_NTL': '{:,.0f}',
                            'Pixel_Count': '{:,.0f}',
                            'Active_Pixels': '{:,.0f}',
                            'Carbon_Per_Pixel': '{:.4f}',
                            'Mean_NTL': '{:.2f}',
                            'Max_NTL': '{:.2f}'
                        }), use_container_width=True)
                        
                        # Download button
                        csv = carbon_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Laporan Karbon (CSV)",
                            data=csv,
                            file_name="carbon_estimation_report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Tidak ada data karbon yang dapat ditampilkan")
                
                elif analysis_type == "Visualisasi Distribusi":
                    st.subheader("🗺️ Visualisasi Distribusi Karbon")
                    
                    selected_idx = st.selectbox(
                        "Pilih dataset untuk visualisasi",
                        options=list(range(len(raster_files))),
                        format_func=lambda x: f"Dataset {x+1} - {raster_files[x].name}"
                    )
                    
                    try:
                        with rasterio.open(raster_paths[selected_idx]) as src:
                            ntl_data = src.read(1)
                            ntl_data[ntl_data == src.nodata] = np.nan
                            
                            # Apply threshold dan hitung karbon
                            filtered_data = apply_threshold_filter(ntl_data, threshold_type, custom_threshold)
                            
                            # Create carbon visualization
                            carbon_coef = custom_coef if conversion_method == "Custom_Linear" else 0.0006
                            carbon_per_pixel = filtered_data * carbon_coef
                            
                            fig = plot_carbon_distribution(
                                ntl_data, 
                                carbon_per_pixel,
                                f"Distribusi Karbon - {raster_files[selected_idx].name}"
                            )
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error dalam visualisasi: {str(e)}")
                
                elif analysis_type == "Analisis Trend":
                    st.subheader("📈 Analisis Trend Karbon")
                    
                    if len(raster_paths) > 1:
                        fig = plot_carbon_trends(carbon_df)
                        st.pyplot(fig)
                    else:
                        st.warning("⚠️ Analisis trend membutuhkan minimal 2 dataset")
                
                elif analysis_type == "Perbandingan Metode":
                    st.subheader("🔬 Perbandingan Metode Konversi")
                    
                    # Bandingkan semua metode
                    methods = ["Elvidge_2009", "Shi_2016", "Power_Law", "Custom_Linear"]
                    comparison_results = []
                    
                    for method in methods:
                        temp_df = generate_carbon_report(raster_paths, method, threshold_type, custom_coef)
                        for idx, row in temp_df.iterrows():
                            comparison_results.append({
                                'Dataset': row['Dataset'],
                                'Method': method,
                                'Carbon_Estimate_Ton': row['Carbon_Estimate_Ton']
                            })
                    
                    comparison_df = pd.DataFrame(comparison_results)
                    
                    # Visualisasi perbandingan
                    if len(comparison_df) > 0:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for method in methods:
                            method_data = comparison_df[comparison_df['Method'] == method]
                            if len(method_data) > 0:
                                ax.plot(range(len(method_data)), method_data['Carbon_Estimate_Ton'], 
                                       'o-', label=method, linewidth=2, markersize=6)
                        
                        ax.set_title('Perbandingan Estimasi Karbon Berbagai Metode')
                        ax.set_xlabel('Dataset')
                        ax.set_ylabel('Estimasi Karbon (Ton)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_xticks(range(len(raster_files)))
                        ax.set_xticklabels([f'DS{i+1}' for i in range(len(raster_files))])
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Tabel perbandingan
                        pivot_df = comparison_df.pivot(index='Dataset', columns='Method', values='Carbon_Estimate_Ton')
                        st.dataframe(pivot_df.style.format('{:,.0f}'))
                    else:
                        st.warning("Tidak ada data untuk perbandingan metode")
                        
            except Exception as e:
                st.error(f"Error dalam menghasilkan laporan karbon: {str(e)}")
    
    else:
        st.info("📁 Silakan unggah file TIFF raster NTL untuk memulai analisis karbon")

# ----------------------------
# FUNGSI UTAMA VISUALISASI GEOSPASIAL - FIXED
# ----------------------------

def setup_geospatial_visualization():
    """Setup utama untuk visualisasi geospasial"""
    
    st.header("🌍 Visualisasi Geospasial Nighttime Lights")
    st.markdown("Analisis spasial dan temporal data nighttime lights dengan visualisasi interaktif")
    
    # Upload data raster
    raster_files = st.file_uploader(
        "Pilih file TIFF raster NTL untuk visualisasi", 
        type=["tif", "tiff"], 
        accept_multiple_files=True,
        help="Unggah beberapa file raster NTL untuk analisis geospasial",
        key="geospatial_upload"
    )
    
    if raster_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_paths = []
            for i, uploaded_file in enumerate(raster_files):
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raster_paths.append(file_path)
            
            st.success(f"✅ {len(raster_paths)} file raster berhasil diunggah")
            
            # Kontrol visualisasi
            col1, col2 = st.columns(2)
            with col1:
                viz_type = st.selectbox(
                    "Tipe Visualisasi",
                    ["Peta Interaktif", "Grid Comparison", "Analisis Statistik", "Single View"]
                )
            
            # Visualisasi berdasarkan pilihan
            if viz_type == "Peta Interaktif":
                st.subheader("🗺️ Peta Interaktif Nighttime Lights")
                years = [f"Data {i+1}" for i in range(len(raster_paths))]
                interactive_map = create_interactive_ntl_map(raster_paths, years)
                if interactive_map:
                    st_folium(interactive_map, width=900, height=600)
                else:
                    st.error("Gagal membuat peta interaktif")
            
            elif viz_type == "Grid Comparison":
                st.subheader("📊 Perbandingan Multi-Temporal")
                titles = [f"Data {i+1} ({uploaded_file.name})" for i, uploaded_file in enumerate(raster_files)]
                comp_fig = plot_ntl_comparison(raster_paths, titles)
                if comp_fig:
                    st.pyplot(comp_fig)
                else:
                    st.error("Gagal membuat visualisasi perbandingan")
            
            elif viz_type == "Analisis Statistik":
                st.subheader("📈 Statistik Spasial NTL")
                
                stats_df = generate_ntl_statistics(raster_paths)
                if len(stats_df) > 0:
                    st.dataframe(stats_df.style.format({
                        'Min': '{:.2f}',
                        'Max': '{:.2f}', 
                        'Mean': '{:.2f}',
                        'Std': '{:.2f}',
                        'Area (px)': '{:,.0f}'
                    }), use_container_width=True)
                    
                    # Visualisasi trend
                    if len(raster_paths) > 1:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        
                        # Data preparation
                        metrics_data = []
                        for path in raster_paths:
                            try:
                                with rasterio.open(path) as src:
                                    data = src.read(1)
                                    data[data == src.nodata] = np.nan
                                    metrics_data.append(data)
                            except:
                                metrics_data.append(np.array([0]))
                        
                        metrics = {
                            'Mean Radiance': [np.nanmean(data) for data in metrics_data],
                            'Max Radiance': [np.nanmax(data) for data in metrics_data],
                            'Illuminated Area': [np.sum(~np.isnan(data)) for data in metrics_data],
                            'Std Dev': [np.nanstd(data) for data in metrics_data]
                        }
                        
                        for idx, (title, values) in enumerate(metrics.items()):
                            ax = axes[idx//2, idx%2]
                            x_range = list(range(len(values)))
                            ax.plot(x_range, values, 'o-', linewidth=2, markersize=6)
                            ax.set_title(title)
                            ax.set_xlabel('Dataset')
                            ax.set_ylabel('Nilai')
                            ax.grid(True, alpha=0.3)
                            ax.set_xticks(x_range)
                            ax.set_xticklabels([f'DS{i+1}' for i in x_range])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.warning("Tidak ada statistik yang dapat ditampilkan")
            
            elif viz_type == "Single View":
                st.subheader("🔍 Detail Visualisasi per Dataset")
                
                selected_idx = st.selectbox(
                    "Pilih dataset",
                    options=list(range(len(raster_files))),
                    format_func=lambda x: f"Dataset {x+1} - {raster_files[x].name}"
                )
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = plot_geospatial_ntl(raster_paths[selected_idx], 
                                            f"Nighttime Lights - {raster_files[selected_idx].name}")
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error("Gagal membuat visualisasi")
                
                with col2:
                    # Statistik dataset terpilih
                    try:
                        with rasterio.open(raster_paths[selected_idx]) as src:
                            data = src.read(1)
                            data[data == src.nodata] = np.nan
                            
                            st.metric("Radiansi Minimum", f"{np.nanmin(data):.2f}")
                            st.metric("Radiansi Maksimum", f"{np.nanmax(data):.2f}")
                            st.metric("Radiansi Rata-rata", f"{np.nanmean(data):.2f}")
                            st.metric("Area Terang (pixels)", f"{np.sum(~np.isnan(data)):,}")
                            
                            # Histogram
                            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                            valid_data = data[~np.isnan(data)]
                            if len(valid_data) > 0:
                                ax_hist.hist(valid_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
                                ax_hist.set_xlabel('Radiansi')
                                ax_hist.set_ylabel('Frekuensi')
                                ax_hist.set_title('Distribusi Radiansi')
                                ax_hist.grid(True, alpha=0.3)
                                st.pyplot(fig_hist)
                            else:
                                st.info("Tidak ada data untuk histogram")
                    except Exception as e:
                        st.error(f"Error dalam menampilkan statistik: {str(e)}")
    else:
        st.info("📁 Silakan unggah file TIFF raster NTL untuk memulai visualisasi")

# ----------------------------
# MAIN APP DENGAN TABS
# ----------------------------

# Tambahkan tabs untuk memisahkan fungsi
tab1, tab2 = st.tabs(["🌿 Analisis Karbon", "🌍 Visualisasi Geospasial"])

with tab1:
    setup_carbon_analysis()

with tab2:
    setup_geospatial_visualization()

# ----------------------------
# INFORMASI DEVELOPER
# ----------------------------
st.markdown("---")
st.markdown("**Nighttime Light (NTL) Carbon Analysis Tool - Ver 5.0**")
st.markdown("**Dikembangkan oleh: Firman Afrianto (NTL Analysis Expert) & Adipandang Yudono (Carbon Estimation Algorithm Creator & WebGIS NTL Analytics Developer)**")
st.markdown("**Fitur: Estimasi Karbon dari Data Nighttime Lights dengan Multiple Conversion Methods**")