# -*- coding: utf-8 -*-
"""
Algoritmo de Processamento para QGIS 3:
Estimativa do Volume de Pastagem e Arbustos por Piquete
Autor: Antigravity / Gemini Pair Programming
"""

import os
import csv
import json
import shutil
import tempfile
import subprocess
import numpy as np
from osgeo import gdal, ogr, osr
from scipy import ndimage

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    QgsApplication,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputVectorLayer,
    QgsProcessingOutputRasterLayer,
    QgsProcessingOutputFile,
    QgsProcessingException,
    QgsVectorLayer,
    QgsFields,
    QgsField,
    QgsFeature,
    QgsGeometry,
    QgsWkbTypes,
    QgsCoordinateTransformContext,
    QgsVectorFileWriter,
    QgsRasterLayer,
    QgsProject,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsPalLayerSettings,
    QgsVectorLayerSimpleLabeling,
    QgsTextFormat,
    QgsTextBufferSettings,
    Qgis
)


class EstimativaVolumePastagemArbustosAlgorithm(QgsProcessingAlgorithm):
    INPUT_LAS = 'INPUT_LAS'
    INPUT_ORTO = 'INPUT_ORTO'
    INPUT_PIQUETES = 'INPUT_PIQUETES'
    RESOLUTION = 'RESOLUTION'
    BUSH_THRESHOLD = 'BUSH_THRESHOLD'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    OUTPUT_PIQUETES = 'OUTPUT_PIQUETES'
    OUTPUT_MHC = 'OUTPUT_MHC'
    OUTPUT_MDS = 'OUTPUT_MDS'
    OUTPUT_MDT = 'OUTPUT_MDT'
    OUTPUT_RASTER_GRASS = 'OUTPUT_RASTER_GRASS'
    OUTPUT_RASTER_BUSH = 'OUTPUT_RASTER_BUSH'
    OUTPUT_BUSH = 'OUTPUT_BUSH'
    OUTPUT_GRASS = 'OUTPUT_GRASS'
    OUTPUT_CSV = 'OUTPUT_CSV'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return EstimativaVolumePastagemArbustosAlgorithm()

    def name(self):
        return 'estimativa_volume_pastagem_arbustos'

    def displayName(self):
        return self.tr('Estimativa do Volume de Pastagem e Arbustos por Piquete')

    def group(self):
        return self.tr('Pastagem e Silvipastoril')

    def groupId(self):
        return 'pastagem_silvipastoril'

    def shortHelpString(self):
        return self.tr(
            """Este algoritmo automatiza o fluxo completo de <b>estimativa volumétrica e biomassa</b> de capim (pastagem) e linhas de arbustos em piquetes:
<br><br>
<b>Entradas:</b>
<ul>
  <li><b>Nuvem de Pontos (*.las, *.laz):</b> Nuvem fotogramétrica ou LiDAR classificada (com Classe 2 para solo).</li>
  <li><b>Ortomosaico RGB (*.tif):</b> Imagem aérea ortorretificada de alta resolução.</li>
  <li><b>Polígonos dos Piquetes (*.shp, *.gpkg):</b> Delimitação individual de cada piquete.</li>
  <li><b>Resolução da Grade:</b> Padrão 0.05 m (5 cm/pixel).</li>
  <li><b>Limiar de Altura dos Arbustos:</b> Padrão 0.70 m.</li>
</ul>
<br>
<b>Produtos Gerados na Pasta de Saída:</b>
<ol>
  <li><code>MDS.tif</code> - Modelo Digital de Superfície</li>
  <li><code>MDT.tif</code> - Modelo Digital do Terreno (Classe 2 Solo interpolada)</li>
  <li><code>MHC.tif</code> - Modelo de Altura do Dossel (MDS - MDT)</li>
  <li><code>raster_altura_capim.tif</code> - Alturas exclusivas da pastagem</li>
  <li><code>raster_altura_arbustos.tif</code> - Alturas exclusivas dos arbustos</li>
  <li><code>poligonos_arbustos.gpkg</code> - Vetor das linhas de arbustos</li>
  <li><code>poligonos_capim.gpkg</code> - Vetor das áreas de pastagem por piquete</li>
  <li><code>piquetes_resultados.gpkg</code> - Vetor dos piquetes com todos os atributos de área, alturas e volumes</li>
  <li><code>tabela_volumes_piquetes.csv</code> - Tabela completa dos resultados por piquete</li>
</ol>"""
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_LAS,
                self.tr('Nuvem de Pontos Classificada (*.las, *.laz)'),
                extension='las'
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_ORTO,
                self.tr('Ortomosaico RGB de Entrada')
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_PIQUETES,
                self.tr('Polígonos dos Piquetes'),
                types=[QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RESOLUTION,
                self.tr('Resolução Espacial da Grade (m/pixel)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.05,
                minValue=0.01,
                maxValue=1.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUSH_THRESHOLD,
                self.tr('Limiar Mínimo de Altura para Arbustos (m)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.70,
                minValue=0.30,
                maxValue=3.0
            )
        )
        param_folder = QgsProcessingParameterFolderDestination(
            self.OUTPUT_FOLDER,
            self.tr('Pasta de Destino dos Produtos Finais (Opcional - deixe vazio para usar temporária)'),
            optional=True
        )
        self.addParameter(param_folder)

        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT_PIQUETES,
                self.tr('Piquetes com Resultados Volumétricos')
            )
        )
        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_MHC,
                self.tr('Modelo de Altura do Dossel (MHC)')
            )
        )
        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_MDS,
                self.tr('Modelo Digital de Superfície (MDS)')
            )
        )
        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_MDT,
                self.tr('Modelo Digital do Terreno (MDT)')
            )
        )
        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_RASTER_GRASS,
                self.tr('Raster de Alturas da Pastagem (Capim)')
            )
        )
        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_RASTER_BUSH,
                self.tr('Raster de Alturas das Linhas de Arbustos')
            )
        )
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT_BUSH,
                self.tr('Polígonos das Linhas de Arbustos')
            )
        )
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT_GRASS,
                self.tr('Polígonos de Pastagem (Capim)')
            )
        )
        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_CSV,
                self.tr('Tabela de Volumes e Alturas (CSV)')
            )
        )

    def find_pdal_executable(self):
        prefix = QgsApplication.prefixPath()
        candidates = [
            os.path.join(prefix, "..", "..", "bin", "pdal.exe"),
            os.path.join(prefix, "..", "bin", "pdal.exe"),
            os.path.join(prefix, "bin", "pdal.exe"),
            r"C:\Program Files\QGIS 3.44.10\bin\pdal.exe",
            r"C:\Program Files\QGIS 3.34\bin\pdal.exe",
            r"C:\OSGeo4W\bin\pdal.exe"
        ]
        for c in candidates:
            cp = os.path.abspath(c)
            if os.path.exists(cp):
                return cp
        which = shutil.which("pdal")
        if which:
            return which
        return "pdal"

    def processAlgorithm(self, parameters, context, feedback):
        las_file = self.parameterAsString(parameters, self.INPUT_LAS, context)
        orto_layer = self.parameterAsRasterLayer(parameters, self.INPUT_ORTO, context)
        piq_layer = self.parameterAsVectorLayer(parameters, self.INPUT_PIQUETES, context)
        res_m = self.parameterAsDouble(parameters, self.RESOLUTION, context)
        if res_m <= 0.0:
            res_m = 0.05
        bush_thresh = self.parameterAsDouble(parameters, self.BUSH_THRESHOLD, context)
        if bush_thresh <= 0.0:
            bush_thresh = 0.70

        out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not out_folder or not os.path.isdir(out_folder):
            out_folder = tempfile.mkdtemp(prefix="volumes_piquetes_")

        os.makedirs(out_folder, exist_ok=True)
        pdal_exe = self.find_pdal_executable()

        feedback.pushInfo(f"Executável PDAL: {pdal_exe}")
        feedback.pushInfo(f"Pasta de saída: {out_folder}")
        feedback.pushInfo(f"Resolução da grade: {res_m:.3f} m | Limiar de arbustos: {bush_thresh:.2f} m")

        # 1. Bounding box from piquetes (ignoring null or empty geometries)
        valid_piq_features = [f for f in piq_layer.getFeatures() if f.hasGeometry() and not f.geometry().isEmpty() and not f.geometry().isNull()]
        if not valid_piq_features:
            raise QgsProcessingException(self.tr("A camada de piquetes não contém geometrias válidas."))

        combined_piq_geom = QgsGeometry.unaryUnion([f.geometry() for f in valid_piq_features])
        ext = combined_piq_geom.boundingBox()
        xmin, xmax = ext.xMinimum() - 5.0, ext.xMaximum() + 5.0
        ymin, ymax = ext.yMinimum() - 5.0, ext.yMaximum() + 5.0
        crs_auth = piq_layer.crs().authid() if piq_layer.crs().isValid() else "EPSG:31983"

        # File paths
        mdt_raw = os.path.join(out_folder, "MDT_raw.tif")
        mds_raw = os.path.join(out_folder, "MDS_raw.tif")
        mdt_final = os.path.join(out_folder, "MDT.tif")
        mds_final = os.path.join(out_folder, "MDS.tif")
        mhc_final = os.path.join(out_folder, "MHC.tif")

        # 2. PDAL Pipeline for MDT (Ground - Class 2)
        feedback.setProgressText(self.tr("1/6: Gerando Modelo Digital do Terreno (MDT) a partir da Classe 2 (Solo)..."))
        feedback.setProgress(10)
        pipe_mdt = {
            "pipeline": [
                {"type": "readers.las", "filename": las_file.replace("\\", "/")},
                {"type": "filters.crop", "bounds": f"([{xmin:.1f}, {xmax:.1f}], [{ymin:.1f}, {ymax:.1f}])"},
                {"type": "filters.range", "limits": "Classification[2:2]"},
                {
                    "type": "writers.gdal",
                    "filename": mdt_raw.replace("\\", "/"),
                    "output_type": "min",
                    "resolution": float(res_m),
                    "window_size": 3,
                    "override_srs": crs_auth
                }
            ]
        }
        self.run_pdal(pdal_exe, pipe_mdt, feedback)

        # 3. PDAL Pipeline for MDS (Surface - Classes 0 and 2)
        feedback.setProgressText(self.tr("2/6: Gerando Modelo Digital de Superfície (MDS) a partir da Vegetação e Solo..."))
        feedback.setProgress(25)
        pipe_mds = {
            "pipeline": [
                {"type": "readers.las", "filename": las_file.replace("\\", "/")},
                {"type": "filters.crop", "bounds": f"([{xmin:.1f}, {xmax:.1f}], [{ymin:.1f}, {ymax:.1f}])"},
                {"type": "filters.range", "limits": "Classification[0:0],Classification[2:2]"},
                {
                    "type": "writers.gdal",
                    "filename": mds_raw.replace("\\", "/"),
                    "output_type": "max",
                    "resolution": float(res_m),
                    "window_size": 3,
                    "override_srs": crs_auth
                }
            ]
        }
        self.run_pdal(pdal_exe, pipe_mds, feedback)

        # 4. Fill NoData & Compute MHC
        feedback.setProgressText(self.tr("3/6: Suavizando terreno e calculando Modelo de Altura do Dossel (MHC)..."))
        feedback.setProgress(40)
        self.fill_nodata_raster(mdt_raw, mdt_final, max_search=30)
        self.fill_nodata_raster(mds_raw, mds_final, max_search=20)
        
        # Remove raw files
        for f in (mdt_raw, mds_raw):
            if os.path.exists(f):
                os.remove(f)

        # Difference MDS - MDT
        ds_s = gdal.Open(mds_final)
        ds_t = gdal.Open(mdt_final)
        gt = ds_s.GetGeoTransform()
        proj = ds_s.GetProjection()
        w = ds_s.RasterXSize
        h = ds_s.RasterYSize
        pixel_area = abs(gt[1] * gt[5])

        arr_s = ds_s.GetRasterBand(1).ReadAsArray()
        arr_t = ds_t.GetRasterBand(1).ReadAsArray()
        valid = (arr_s != -9999.0) & (arr_t != -9999.0) & ~np.isnan(arr_s) & ~np.isnan(arr_t)
        diff = np.where(valid, arr_s - arr_t, -9999.0)
        mhc_arr = np.where(valid, np.clip(diff, 0.0, 25.0), -9999.0).astype(np.float32)

        drv = gdal.GetDriverByName("GTiff")
        ds_mhc = drv.Create(mhc_final, w, h, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=2"])
        ds_mhc.SetGeoTransform(gt)
        ds_mhc.SetProjection(proj)
        ds_mhc.GetRasterBand(1).SetNoDataValue(-9999.0)
        ds_mhc.GetRasterBand(1).WriteArray(mhc_arr)
        ds_mhc.BuildOverviews("AVERAGE", [2, 4, 8, 16])
        ds_mhc = None
        ds_s = None
        ds_t = None

        # 5. Segment Bush and Grass
        feedback.setProgressText(self.tr("4/6: Segmentando e vetorizando linhas de arbustos e pastagem..."))
        feedback.setProgress(60)

        bush_binary = (mhc_arr >= bush_thresh)
        struct = ndimage.generate_binary_structure(2, 2)
        bush_clean = ndimage.binary_opening(bush_binary, structure=struct, iterations=1)
        bush_clean = ndimage.binary_closing(bush_clean, structure=struct, iterations=2)
        lbl, n_feat = ndimage.label(bush_clean, structure=struct)
        sizes = ndimage.sum(bush_clean, lbl, range(n_feat + 1))
        min_pixels = int(0.20 / pixel_area)
        mask_size = sizes >= min_pixels
        bush_mask = mask_size[lbl]

        # Rasterize piquetes
        mem_drv = gdal.GetDriverByName("MEM")
        target_ds = mem_drv.Create("", w, h, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(gt)
        target_ds.SetProjection(proj)

        ogr_ds = ogr.Open(piq_layer.source())
        ogr_lyr = ogr_ds.GetLayer()
        gdal.RasterizeLayer(target_ds, [1], ogr_lyr, burn_values=[1])
        piq_mask = target_ds.GetRasterBand(1).ReadAsArray() == 1
        target_ds = None

        bush_in_piq = bush_mask & piq_mask
        grass_in_piq = piq_mask & (~bush_in_piq)

        # Export isolated rasters
        r_grass_path = os.path.join(out_folder, "raster_altura_capim.tif")
        r_bush_path = os.path.join(out_folder, "raster_altura_arbustos.tif")

        ds_g = drv.Create(r_grass_path, w, h, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
        ds_g.SetGeoTransform(gt)
        ds_g.SetProjection(proj)
        ds_g.GetRasterBand(1).SetNoDataValue(-9999.0)
        ds_g.GetRasterBand(1).WriteArray(np.where(grass_in_piq, mhc_arr, -9999.0).astype(np.float32))
        ds_g = None

        ds_b = drv.Create(r_bush_path, w, h, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
        ds_b.SetGeoTransform(gt)
        ds_b.SetProjection(proj)
        ds_b.GetRasterBand(1).SetNoDataValue(-9999.0)
        ds_b.GetRasterBand(1).WriteArray(np.where(bush_in_piq, mhc_arr, -9999.0).astype(np.float32))
        ds_b = None

        # Vectorize Bush to GPKG
        bush_mask_tif = os.path.join(out_folder, "mask_arbustos.tif")
        ds_m = drv.Create(bush_mask_tif, w, h, 1, gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
        ds_m.SetGeoTransform(gt)
        ds_m.SetProjection(proj)
        ds_m.GetRasterBand(1).WriteArray(bush_in_piq.astype(np.uint8))
        ds_m = None

        bush_gpkg = os.path.join(out_folder, "poligonos_arbustos.gpkg")
        if os.path.exists(bush_gpkg):
            os.remove(bush_gpkg)

        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        gpkg_drv = ogr.GetDriverByName("GPKG")
        dst_ds = gpkg_drv.CreateDataSource(bush_gpkg)
        dst_lyr = dst_ds.CreateLayer("poligonos_arbustos", srs=srs, geom_type=ogr.wkbPolygon)
        fd = ogr.FieldDefn("DN", ogr.OFTInteger)
        dst_lyr.CreateField(fd)
        
        src_mask_ds = gdal.Open(bush_mask_tif)
        gdal.Polygonize(src_mask_ds.GetRasterBand(1), None, dst_lyr, 0, [], callback=None)
        dst_ds.ExecuteSQL("DELETE FROM poligonos_arbustos WHERE DN = 0")
        dst_ds = None
        src_mask_ds = None

        # Create poligonos_capim.gpkg by difference
        lyr_bush = QgsVectorLayer(bush_gpkg, "bush", "ogr")
        bush_geoms = [f.geometry() for f in lyr_bush.getFeatures() if f.hasGeometry()]
        combined_bush = QgsGeometry.unaryUnion(bush_geoms) if bush_geoms else QgsGeometry()

        grass_gpkg = os.path.join(out_folder, "poligonos_capim.gpkg")
        if os.path.exists(grass_gpkg):
            os.remove(grass_gpkg)

        grass_fields = QgsFields()
        grass_fields.append(QgsField("Piquete_ID", QVariant.Int))
        grass_fields.append(QgsField("Area_m2", QVariant.Double))

        options_g = QgsVectorFileWriter.SaveVectorOptions()
        options_g.driverName = "GPKG"
        options_g.layerName = "poligonos_capim"

        writer_g = QgsVectorFileWriter.create(
            grass_gpkg, grass_fields, QgsWkbTypes.MultiPolygon, piq_layer.crs(),
            QgsCoordinateTransformContext(), options_g
        )
        for idx, f in enumerate(valid_piq_features):
            geom = f.geometry()
            diff_geom = geom.difference(combined_bush) if not combined_bush.isEmpty() else geom
            if not diff_geom.isMultipart():
                diff_geom.convertToMultiType()
            new_f = QgsFeature(grass_fields)
            new_f.setAttributes([idx + 1, diff_geom.area()])
            new_f.setGeometry(diff_geom)
            writer_g.addFeature(new_f)
        del writer_g

        # 6. Zonal Statistics and Volume Calculation
        feedback.setProgressText(self.tr("5/6: Calculando volumes e métricas por piquete..."))
        feedback.setProgress(80)

        # Sort valid features logically (North-to-South, West-to-East)
        def sort_key(f):
            c = f.geometry().centroid().asPoint()
            return (-c.y(), c.x())

        piq_sorted = sorted(valid_piq_features, key=sort_key)

        results = []
        for idx, f in enumerate(piq_sorted):
            p_id = idx + 1
            geom = f.geometry()
            
            # Rasterize individual paddock
            mem_target = mem_drv.Create("", w, h, 1, gdal.GDT_Byte)
            mem_target.SetGeoTransform(gt)
            mem_target.SetProjection(proj)

            mem_ogr = ogr.GetDriverByName("Memory").CreateDataSource("")
            mem_lyr = mem_ogr.CreateLayer("single_p", srs=ogr_lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon)
            feat_copy = ogr.Feature(mem_lyr.GetLayerDefn())
            feat_copy.SetGeometry(ogr.CreateGeometryFromWkt(geom.asWkt()))
            mem_lyr.CreateFeature(feat_copy)

            gdal.RasterizeLayer(mem_target, [1], mem_lyr, burn_values=[1])
            single_mask = mem_target.GetRasterBand(1).ReadAsArray() == 1
            mem_target = None
            mem_ogr = None

            mask_p_bush = single_mask & bush_mask
            mask_p_grass = single_mask & (~bush_mask)

            # Grass
            p_grass_pix = mhc_arr[mask_p_grass]
            p_grass_valid = p_grass_pix[p_grass_pix >= 0]
            g_area = len(p_grass_valid) * pixel_area
            g_mean = float(np.mean(p_grass_valid)) if len(p_grass_valid) > 0 else 0.0
            g_max = float(np.max(p_grass_valid)) if len(p_grass_valid) > 0 else 0.0
            g_vol = float(np.sum(p_grass_valid) * pixel_area) if len(p_grass_valid) > 0 else 0.0

            # Bush
            p_bush_pix = mhc_arr[mask_p_bush]
            p_bush_valid = p_bush_pix[p_bush_pix >= 0]
            b_area = len(p_bush_valid) * pixel_area
            b_mean = float(np.mean(p_bush_valid)) if len(p_bush_valid) > 0 else 0.0
            b_max = float(np.max(p_bush_valid)) if len(p_bush_valid) > 0 else 0.0
            b_vol = float(np.sum(p_bush_valid) * pixel_area) if len(p_bush_valid) > 0 else 0.0

            t_area = g_area + b_area
            results.append({
                'Piquete_ID': p_id,
                'Identificacao': f"Piquete {p_id:02d}",
                'Area_Total_m2': round(t_area, 2),
                'Area_Capim_m2': round(g_area, 2),
                'Area_Arbustos_m2': round(b_area, 2),
                'Perc_Capim': round((g_area / t_area) * 100, 1) if t_area > 0 else 0.0,
                'Perc_Arbustos': round((b_area / t_area) * 100, 1) if t_area > 0 else 0.0,
                'Altura_Media_Capim_m': round(g_mean, 3),
                'Altura_Maxima_Capim_m': round(g_max, 3),
                'Volume_Capim_m3': round(g_vol, 2),
                'Altura_Media_Arbustos_m': round(b_mean, 3),
                'Altura_Maxima_Arbustos_m': round(b_max, 3),
                'Volume_Arbustos_m3': round(b_vol, 2),
                'geom': geom
            })

        # Save CSV
        csv_out = os.path.join(out_folder, "tabela_volumes_piquetes.csv")
        csv_keys = [k for k in results[0].keys() if k != 'geom']
        with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
            cw = csv.DictWriter(f, fieldnames=csv_keys, delimiter=';')
            cw.writeheader()
            for r in results:
                row = {k: r[k] for k in csv_keys}
                cw.writerow(row)

        # Save Vector GPKG
        piq_gpkg = os.path.join(out_folder, "piquetes_resultados.gpkg")
        if os.path.exists(piq_gpkg):
            os.remove(piq_gpkg)

        res_fields = QgsFields()
        res_fields.append(QgsField("Piquete_ID", QVariant.Int))
        res_fields.append(QgsField("Identificacao", QVariant.String))
        res_fields.append(QgsField("Area_Total_m2", QVariant.Double))
        res_fields.append(QgsField("Area_Capim_m2", QVariant.Double))
        res_fields.append(QgsField("Area_Arbustos_m2", QVariant.Double))
        res_fields.append(QgsField("Perc_Capim", QVariant.Double))
        res_fields.append(QgsField("Perc_Arbustos", QVariant.Double))
        res_fields.append(QgsField("H_Med_Capim_m", QVariant.Double))
        res_fields.append(QgsField("H_Max_Capim_m", QVariant.Double))
        res_fields.append(QgsField("Vol_Capim_m3", QVariant.Double))
        res_fields.append(QgsField("H_Med_Arb_m", QVariant.Double))
        res_fields.append(QgsField("H_Max_Arb_m", QVariant.Double))
        res_fields.append(QgsField("Vol_Arbustos_m3", QVariant.Double))

        options_p = QgsVectorFileWriter.SaveVectorOptions()
        options_p.driverName = "GPKG"
        options_p.layerName = "piquetes_resultados"

        writer_p = QgsVectorFileWriter.create(
            piq_gpkg, res_fields, QgsWkbTypes.Polygon, piq_layer.crs(),
            QgsCoordinateTransformContext(), options_p
        )
        for r in results:
            feat_res = QgsFeature(res_fields)
            feat_res.setAttributes([
                r['Piquete_ID'], r['Identificacao'], r['Area_Total_m2'],
                r['Area_Capim_m2'], r['Area_Arbustos_m2'], r['Perc_Capim'], r['Perc_Arbustos'],
                r['Altura_Media_Capim_m'], r['Altura_Maxima_Capim_m'], r['Volume_Capim_m3'],
                r['Altura_Media_Arbustos_m'], r['Altura_Maxima_Arbustos_m'], r['Volume_Arbustos_m3']
            ])
            feat_res.setGeometry(r['geom'])
            writer_p.addFeature(feat_res)
        del writer_p

        feedback.setProgressText(self.tr("6/6: Carregando e estilizando camadas no projeto QGIS..."))
        feedback.setProgress(95)

        # Automatically load and style layers into the current QGIS project if running in GUI
        if context.project():
            # 1. Carrega MDS e MDT
            mds_layer = QgsRasterLayer(mds_final, "Modelo Digital de Superfície (MDS)")
            if mds_layer.isValid():
                context.project().addMapLayer(mds_layer)

            mdt_layer = QgsRasterLayer(mdt_final, "Modelo Digital do Terreno (MDT)")
            if mdt_layer.isValid():
                context.project().addMapLayer(mdt_layer)

            # 2. Carrega MHC
            mhc_layer = QgsRasterLayer(mhc_final, "Modelo Digital de Altura (MHC)")
            if mhc_layer.isValid():
                context.project().addMapLayer(mhc_layer)

            # 3. Carrega Rasters de Altura Isolados (Capim e Arbustos)
            r_grass_layer = QgsRasterLayer(r_grass_path, "Raster Altura - Pastagem (Capim)")
            if r_grass_layer.isValid():
                context.project().addMapLayer(r_grass_layer)

            r_bush_layer = QgsRasterLayer(r_bush_path, "Raster Altura - Linhas de Arbustos")
            if r_bush_layer.isValid():
                context.project().addMapLayer(r_bush_layer)

            # 4. Carrega Polígonos de Pastagem e Arbustos
            grass_layer = QgsVectorLayer(grass_gpkg, "Polígonos de Pastagem", "ogr")
            if grass_layer.isValid():
                sym_g = QgsFillSymbol.createSimple({
                    'color': '144,238,144,100',
                    'outline_color': '34,139,34',
                    'outline_width': '0.2'
                })
                grass_layer.setRenderer(QgsSingleSymbolRenderer(sym_g))
                context.project().addMapLayer(grass_layer)

            bush_layer = QgsVectorLayer(bush_gpkg, "Linhas de Arbustos", "ogr")
            if bush_layer.isValid():
                sym_b = QgsFillSymbol.createSimple({
                    'color': '34,139,34,180',
                    'outline_color': '0,100,0',
                    'outline_width': '0.3'
                })
                bush_layer.setRenderer(QgsSingleSymbolRenderer(sym_b))
                context.project().addMapLayer(bush_layer)

            # 5. Carrega Piquetes com Resultados Volumétricos (Bordas verdes + Rótulos completos)
            res_layer = QgsVectorLayer(piq_gpkg, "Piquetes - Volumes e Alturas", "ogr")
            if res_layer.isValid():
                sym_p = QgsFillSymbol.createSimple({
                    'color': '0,0,0,0',
                    'outline_color': '0,180,50',
                    'outline_width': '0.7',
                    'outline_style': 'solid'
                })
                res_layer.setRenderer(QgsSingleSymbolRenderer(sym_p))

                lbl = QgsPalLayerSettings()
                lbl.fieldName = """'Piquete ' || lpad(Piquete_ID, 2, '0') || '\n' ||
'Pasto: ' || round(Vol_Capim_m3, 0) || ' m³\n' ||
'Arbustos: ' || round(Vol_Arbustos_m3, 0) || ' m³'"""
                lbl.isExpression = True
                lbl.placement = Qgis.LabelPlacement.Horizontal
                lbl.fitInPolygonOnly = False

                t_fmt = QgsTextFormat()
                t_fmt.setFont(QFont("Arial", 9, QFont.Bold))
                t_fmt.setColor(QColor(15, 15, 15))

                t_buf = QgsTextBufferSettings()
                t_buf.setEnabled(True)
                t_buf.setSize(1.2)
                t_buf.setColor(QColor(255, 255, 255))
                t_fmt.setBuffer(t_buf)

                lbl.setFormat(t_fmt)
                res_layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
                res_layer.setLabelsEnabled(True)

                context.project().addMapLayer(res_layer)

        feedback.setProgress(100)
        feedback.pushInfo(f"Processamento finalizado com sucesso! Resultados em: {out_folder}")

        return {
            self.OUTPUT_PIQUETES: piq_gpkg,
            self.OUTPUT_MHC: mhc_final,
            self.OUTPUT_MDS: mds_final,
            self.OUTPUT_MDT: mdt_final,
            self.OUTPUT_RASTER_GRASS: r_grass_path,
            self.OUTPUT_RASTER_BUSH: r_bush_path,
            self.OUTPUT_BUSH: bush_gpkg,
            self.OUTPUT_GRASS: grass_gpkg,
            self.OUTPUT_CSV: csv_out
        }

    def run_pdal(self, pdal_exe, pipeline_dict, feedback):
        pipe_json = json.dumps(pipeline_dict, indent=2)
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        proc = subprocess.Popen(
            [pdal_exe, "pipeline", "--stdin"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            startupinfo=startupinfo
        )
        stdout, stderr = proc.communicate(input=pipe_json)
        if proc.returncode != 0:
            raise QgsProcessingException(f"Erro na execução do PDAL:\n{stderr}\n{stdout}")

    def fill_nodata_raster(self, src_path, dst_path, max_search=25):
        src_ds = gdal.Open(src_path)
        gt = src_ds.GetGeoTransform()
        proj = src_ds.GetProjection()
        w = src_ds.RasterXSize
        h = src_ds.RasterYSize
        band = src_ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        drv = gdal.GetDriverByName("GTiff")
        dst_ds = drv.Create(dst_path, w, h, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=2"])
        dst_ds.SetGeoTransform(gt)
        dst_ds.SetProjection(proj)
        out_band = dst_ds.GetRasterBand(1)
        out_band.SetNoDataValue(nodata)
        out_band.WriteArray(band.ReadAsArray())
        gdal.FillNodata(targetBand=out_band, maskBand=None, maxSearchDist=max_search, smoothingIterations=1)
        dst_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16])
        dst_ds = None
        src_ds = None
