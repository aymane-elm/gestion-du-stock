import os
import io
import uuid
import tempfile
from datetime import datetime, date, timedelta
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
DEFAULT_EXCEL_PATH = os.getenv("INVENTORY_XLSX", "inventory.xlsx")
REQUIRED_SHEETS = ["Stock","Mouvements","Fabrications","BOM_GMQ_ONE","BOM_GMQ_LIVE"]
OPTIONAL_SHEETS = ["Responsables","Clients"]  # créées si manquantes

st.set_page_config(page_title="Stock & Fabrication", layout="wide")
st.title("Gestion de stock & fabrication")

# =========================
# SCHEMAS (types attendus)
# =========================
SCHEMA: Dict[str, Dict[str, object]] = {
    "Stock": {
        "SKU": "object", "Name": "object", "Unit": "object", "Category": "object",
        "ReorderPoint": "float64", "QtyOnHand": "float64", "Description": "object"
    },
    # Client retiré de Mouvements
    "Mouvements": {
        "Date": "object", "SKU": "object", "Type": "object", "Qty": "float64",
        "Ref": "object", "Location": "object", "MO_ID": "object",
        "Responsable": "object"
    },
    # Client ajouté dans Fabrications
    "Fabrications": {
        "MO_ID": "object", "Date": "object", "DueDate": "object", "Product": "object",
        "Qty": "float64", "Status": "object", "Ref": "object",
        "Responsable": "object", "Client": "object"
    },
    "BOM_GMQ_ONE": {"ComponentSKU": "object", "QtyPerUnit": "float64", "Description": "object"},
    "BOM_GMQ_LIVE": {"ComponentSKU": "object", "QtyPerUnit": "float64", "Description": "object"},
    "Responsables": {"Responsable": "object"},
    "Clients": {
        "ClientID": "object", "ClientName": "object", "Type": "object",
        "Phone": "object", "Email": "object", "Notes": "object"
    },
}

# =========================
# HELPERS I/O
# =========================
def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.isfile(path)
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def read_excel_from_path(path: str) -> Dict[str, pd.DataFrame]:
    if not file_exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    xls = pd.ExcelFile(path, engine="openpyxl")
    missing = [s for s in REQUIRED_SHEETS if s not in xls.sheet_names]
    if missing:
        raise ValueError(f"Feuilles manquantes dans le fichier : {missing}")
    dfs: Dict[str, pd.DataFrame] = {s: xls.parse(s) for s in REQUIRED_SHEETS if s in xls.sheet_names}
    for s in OPTIONAL_SHEETS:
        if s in xls.sheet_names:
            dfs[s] = xls.parse(s)
    # défauts si absents
    if "Responsables" not in dfs:
        dfs["Responsables"] = pd.DataFrame({"Responsable": ["Aymane","Joslain","Lise","Robin"]})
    if "Clients" not in dfs:
        dfs["Clients"] = pd.DataFrame({
            "ClientID": [], "ClientName": [], "Type": [], "Phone": [], "Email": [], "Notes": []
        })
    # Normalisation stricte selon SCHEMA
    for name, schema in SCHEMA.items():
        df = dfs.get(name, pd.DataFrame())
        for col, dtype in schema.items():
            if col not in df.columns:
                df[col] = np.nan if dtype != "object" else None
        df = df[[c for c in schema.keys()]]  # ordre + supprime colonnes inconnues
        for col, dtype in schema.items():
            if dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            else:
                df[col] = df[col].astype("object")
        dfs[name] = df
    return dfs

def _write_excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name in REQUIRED_SHEETS + OPTIONAL_SHEETS:
            df = dfs.get(name, pd.DataFrame())
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return buf.read()

def write_excel_to_path_atomic(dfs: Dict[str, pd.DataFrame], path: str) -> None:
    payload = _write_excel_bytes(dfs)
    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".xlsx") as tmp:
        tmp.write(payload)
        tmp.flush()
        tmp_name = tmp.name
    try:
        os.replace(tmp_name, path)
    except PermissionError as e:
        try: os.remove(tmp_name)
        except Exception: pass
        raise PermissionError("Impossible d'écrire le fichier. Ferme-le dans Excel puis réessaie.") from e

# =========================
# LOGIQUE STOCK / MOUVEMENTS / OF
# =========================
@st.cache_data(show_spinner=False)
def build_stock_index(stock: pd.DataFrame) -> Dict[str, int]:
    return {sku: i for i, sku in enumerate(stock["SKU"].astype(str).tolist())}

def apply_movement(stock: pd.DataFrame, stock_idx: Dict[str, int], sku: str, qty: float, move_type: str) -> Tuple[pd.DataFrame, float]:
    if qty < 0:
        raise ValueError("La quantité doit être >= 0.")
    key = str(sku)
    if key not in stock_idx:
        raise ValueError(f"SKU introuvable: {sku}")
    i = stock_idx[key]
    current = float(stock.at[i, "QtyOnHand"]) if not pd.isna(stock.at[i, "QtyOnHand"]) else 0.0
    delta = qty if move_type == "IN" else -qty
    new_val = current + delta
    stock.at[i, "QtyOnHand"] = new_val
    return stock, new_val

def record_move(movements: pd.DataFrame, sku: str, move_type: str, qty: float,
                ref: str, location: str, mo_id: str | None, responsable: str | None) -> pd.DataFrame:
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "SKU": str(sku), "Type": move_type, "Qty": float(qty),
        "Ref": ref, "Location": location, "MO_ID": mo_id,
        "Responsable": responsable
    }
    return pd.concat([movements, pd.DataFrame([row])], ignore_index=True)

def load_bom(dfs: Dict[str, pd.DataFrame], product: str) -> pd.DataFrame:
    if product == "GMQ ONE":
        return dfs["BOM_GMQ_ONE"][["ComponentSKU","QtyPerUnit","Description"]].copy()
    elif product == "GMQ LIVE":
        return dfs["BOM_GMQ_LIVE"][["ComponentSKU","QtyPerUnit","Description"]].copy()
    else:
        raise ValueError("Produit inconnu (attendu: GMQ ONE / GMQ LIVE)")

def check_availability(dfs: Dict[str, pd.DataFrame], product: str, qty_make: float) -> Tuple[pd.DataFrame, bool]:
    bom = load_bom(dfs, product)
    bom["ComponentSKU"] = bom["ComponentSKU"].astype(str)
    stock = dfs["Stock"][["SKU","QtyOnHand"]].copy()
    stock["SKU"] = stock["SKU"].astype(str)

    need_df = bom.assign(**{"Besoin (total)": bom["QtyPerUnit"].fillna(0.0) * float(qty_make or 0)})
    merged = need_df.merge(stock, left_on="ComponentSKU", right_on="SKU", how="left")
    merged["QtyOnHand"].fillna(0.0, inplace=True)
    merged.rename(columns={"QtyOnHand": "Stock dispo"}, inplace=True)
    merged["Manque"] = (merged["Besoin (total)"] - merged["Stock dispo"]).clip(lower=0.0)

    out_cols = ["ComponentSKU","QtyPerUnit","Besoin (total)","Stock dispo","Manque","Description"]
    df_req = merged[out_cols]
    ok = float(df_req["Manque"].sum()) == 0.0
    df_req = df_req.sort_values(["Manque","ComponentSKU"], ascending=[False,True]).reset_index(drop=True)
    return df_req, ok

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# =========================
# SIDEBAR / CHARGEMENT
# =========================
st.sidebar.header("Fichier Excel")
excel_path = st.sidebar.text_input("Chemin du fichier", value=DEFAULT_EXCEL_PATH)
st.sidebar.caption("Tu peux changer le chemin à la volée.")
col_sb1, col_sb2 = st.sidebar.columns(2)

# Recharger : reset cache + session puis rerun
if col_sb1.button("Recharger"):
    st.cache_data.clear()
    st.session_state.pop("dfs", None)
    st.session_state.pop("excel_loaded_from", None)
    st.rerun()

autosave = st.sidebar.toggle("Sauvegarde auto", value=True, help="Écrit après chaque transaction (déconseillé sur partage réseau instable)")

# Lecture disque
try:
    dfs_loaded = read_excel_from_path(excel_path)
except Exception as e:
    st.error(f"Problème de chargement : {e}")
    st.stop()

# Initialisation session
if "dfs" not in st.session_state:
    st.session_state.dfs = dfs_loaded
    st.session_state.excel_loaded_from = excel_path
else:
    if st.session_state.get("excel_loaded_from") != excel_path:
        st.session_state.dfs = dfs_loaded
        st.session_state.excel_loaded_from = excel_path

dfs = st.session_state.dfs

# Index SKU → ligne
stock_index = build_stock_index(dfs["Stock"])

if col_sb2.button("Sauvegarder maintenant"):
    try:
        write_excel_to_path_atomic(dfs, excel_path)
        st.sidebar.success("Fichier sauvegardé")
    except PermissionError as e:
        st.sidebar.error(str(e))

# =========================
# UI – TABS
# =========================
tab_dash, tab_moves, tab_mo, tab_stock, tab_compos, tab_invent, tab_clients, tab_export = st.tabs([
    "Dashboard","Mouvements","Ordres de fabrication","Stock",
    "Composants","Inventaire","Clients","Export CSV"
])

# ---- DASHBOARD
with tab_dash:
    st.subheader("Aperçu")
    col1, col2, col3, col4 = st.columns(4)
    total_skus = int(len(dfs["Stock"]))
    total_qty = float(dfs["Stock"]["QtyOnHand"].sum())
    mo_posted = int((dfs["Fabrications"]["Status"] == "Posté").sum()) if "Status" in dfs["Fabrications"].columns else 0
    low = dfs["Stock"][ dfs["Stock"]["QtyOnHand"] < dfs["Stock"]["ReorderPoint"] ] if "ReorderPoint" in dfs["Stock"].columns else pd.DataFrame()
    col1.metric("Articles (SKU)", f"{total_skus}")
    col2.metric("Qté totale", f"{total_qty:,.0f}")
    col3.metric("OF postés", f"{mo_posted}")
    col4.metric("Sous seuil", f"{len(low)}")
    st.markdown("#### Sous le seuil (selon *ReorderPoint*)")
    st.dataframe(low if not low.empty else pd.DataFrame(columns=dfs["Stock"].columns), use_container_width=True)

# ---- MOUVEMENTS (form + filtres + tableau)
with tab_moves:
    st.header("Mouvements")

    # --- Saisie d’un mouvement
    st.subheader("Ajouter un mouvement")
    resp_list = dfs["Responsables"]["Responsable"].dropna().astype(str).tolist() or ["Aymane","Joslain","Lise","Robin"]
    with st.form("mv_form"):
        col_a, col_b = st.columns(2)
        sku = col_a.selectbox("SKU", dfs["Stock"]["SKU"].astype(str).tolist(), key="mv_sku")
        responsable = col_b.selectbox("Responsable", resp_list, index=0, key="mv_resp")
        move_type = st.radio("Type", ["IN","OUT"], horizontal=True, key="mv_type")
        qty = st.number_input("Quantité", min_value=0.0, step=1.0, key="mv_qty")
        ref = st.text_input("Référence", value="MANUAL", key="mv_ref")
        loc = st.text_input("Emplacement", value="ENTREPOT", key="mv_loc")
        submitted = st.form_submit_button("Enregistrer")
        if submitted:
            if qty <= 0:
                st.error("La quantité doit être > 0.")
            else:
                try:
                    dfs["Stock"], new_qty = apply_movement(dfs["Stock"], stock_index, sku, qty, move_type)
                    dfs["Mouvements"] = record_move(dfs["Mouvements"], sku, move_type, qty, ref, loc, mo_id=None, responsable=responsable)
                    if autosave:
                        write_excel_to_path_atomic(dfs, excel_path)
                    st.success(f"Mouvement {move_type} enregistré par {responsable}. Nouveau stock {sku} = {new_qty}")
                    st.toast("Mouvement enregistré")
                except PermissionError as e:
                    st.error(str(e))

    st.divider()

    # --- Filtres + tableau mouvements
    st.subheader("Historique des mouvements")
    mv = dfs["Mouvements"].copy()
    mv["Date_dt"] = pd.to_datetime(mv["Date"], errors="coerce")
    min_d = pd.to_datetime(mv["Date_dt"].min()).date() if not mv["Date_dt"].isna().all() else date.today() - timedelta(days=30)
    max_d = pd.to_datetime(mv["Date_dt"].max()).date() if not mv["Date_dt"].isna().all() else date.today()

    c1, c2, c3 = st.columns(3)
    d_from = c1.date_input("Du", value=min_d, key="mv_from")
    d_to   = c2.date_input("Au", value=max_d, key="mv_to")
    types = c3.multiselect("Type", options=["IN","OUT"], default=["IN","OUT"], key="mv_types")

    c4, c5 = st.columns(2)
    sku_filter = c4.text_input("Filtre SKU (contient)", "", key="mv_sku_filter")
    resp_opts = ["(Tous)"] + sorted(mv["Responsable"].dropna().astype(str).unique().tolist())
    resp_pick = c5.selectbox("Responsable", resp_opts, index=0, key="mv_resp_filter")

    mv_view = mv.drop(columns=["Date_dt"]).copy()
    mask_date = (mv["Date_dt"].dt.date >= d_from) & (mv["Date_dt"].dt.date <= d_to)
    mv_view = mv_view[mask_date]
    mv_view = mv_view[mv["Type"].isin(types)]
    if sku_filter.strip():
        mv_view = mv_view[mv_view["SKU"].astype(str).str.contains(sku_filter, case=False, na=False)]
    if resp_pick != "(Tous)":
        mv_view = mv_view[mv_view["Responsable"].astype(str) == resp_pick]

    st.dataframe(mv_view.sort_values("Date", ascending=False), use_container_width=True)

# ---- ORDRES DE FABRICATION (vérif + création + filtres + tableau)
with tab_mo:
    st.header("Ordres de fabrication")

    # --- Création / Vérification
    st.subheader("Créer un OF")
    resp_list = dfs["Responsables"]["Responsable"].dropna().astype(str).tolist() or ["Aymane","Joslain","Lise","Robin"]
    clients_list = dfs["Clients"]["ClientName"].dropna().astype(str).tolist()

    with st.form("mo_form"):
        col1, col2 = st.columns(2)
        product = col1.selectbox("Produit fini", ["GMQ ONE","GMQ LIVE"], key="mo_product")
        responsable = col2.selectbox("Responsable", resp_list, index=0, key="mo_resp")

        col3, col4, col5 = st.columns([1,1,2])
        qty_make = col3.number_input("Quantité à produire", min_value=0.0, step=1.0, key="mo_qty")
        due_date = col4.date_input("Date d'échéance", value=date.today() + timedelta(days=7), key="mo_due")
        ref = col5.text_input("Référence OF", value="OF-AUTO", key="mo_ref")

        st.markdown("**Client associé à l'OF**")
        colc1, colc2 = st.columns(2)
        client_pick = ["(aucun)"] + clients_list + ["Client de passage (saisie)"]
        client_choice = colc1.selectbox("Client", client_pick, index=0, key="mo_client_pick")
        client_free = None
        if client_choice == "Client de passage (saisie)":
            client_free = colc2.text_input("Nom du client (passage)", value="", key="mo_client_free")

        cver, cpost = st.columns(2)
        verify_clicked = cver.form_submit_button("Vérifier l'OF")
        post_clicked = cpost.form_submit_button("Poster l'OF")

        if verify_clicked or post_clicked:
            if qty_make <= 0:
                st.error("La quantité doit être > 0.")
            else:
                req_df, ok = check_availability(dfs, product, qty_make)
                st.markdown("#### Besoins vs stock (BOM)")
                st.dataframe(req_df, use_container_width=True)

                if not ok:
                    manques = req_df.loc[req_df["Manque"] > 0, ["ComponentSKU","Manque"]]
                    manques_str = ", ".join([f"{r.ComponentSKU} (-{r.Manque:.0f})" if float(r.Manque).is_integer() else f"{r.ComponentSKU} (-{r.Manque})" for r in manques.itertuples()])
                    st.error(f"Stock insuffisant. Compléter le stock avant de poster. Composants manquants : {manques_str}")
                else:
                    st.success("Stock OK pour l'OF.")

                if post_clicked and ok:
                    try:
                        client_final = None
                        if client_choice == "Client de passage (saisie)":
                            client_final = (client_free or "").strip() or None
                        elif client_choice not in ["(aucun)", "Client de passage (saisie)"]:
                            client_final = client_choice

                        mo_id = uuid.uuid4().hex[:8].upper()
                        mo_row = {
                            "MO_ID": mo_id,
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Product": product,
                            "Qty": float(qty_make),
                            "Status": "Posté",
                            "Ref": ref,
                            "DueDate": due_date.strftime("%Y-%m-%d"),
                            "Responsable": responsable,
                            "Client": client_final,
                        }
                        dfs["Fabrications"] = pd.concat([dfs["Fabrications"], pd.DataFrame([mo_row])], ignore_index=True)

                        # Déductions OUT des composants
                        for _, r in req_df.iterrows():
                            comp_sku = r["ComponentSKU"]
                            need = float(r["Besoin (total)"])
                            dfs["Stock"], _ = apply_movement(dfs["Stock"], stock_index, comp_sku, need, "OUT")
                            dfs["Mouvements"] = record_move(dfs["Mouvements"], comp_sku, "OUT", need, ref, "PROD", mo_id, responsable)

                        # Entrée du produit fini
                        fin_sku = "GMQ-ONE" if product == "GMQ ONE" else "GMQ-LIVE"
                        dfs["Stock"], _ = apply_movement(dfs["Stock"], stock_index, fin_sku, qty_make, "IN")
                        dfs["Mouvements"] = record_move(dfs["Mouvements"], fin_sku, "IN", qty_make, ref, "STOCK", mo_id, responsable)

                        if autosave:
                            write_excel_to_path_atomic(dfs, excel_path)
                        st.success(f"OF {mo_id} posté par {responsable} (échéance {due_date.strftime('%Y-%m-%d')}).")
                        st.toast("OF posté")
                    except PermissionError as e:
                        st.error(str(e))

    st.divider()

    # --- Filtres + tableau OF
    st.subheader("Liste des ordres de fabrication")
    fab = dfs["Fabrications"].copy()
    fab["Date_dt"] = pd.to_datetime(fab["Date"], errors="coerce")
    min_f = pd.to_datetime(fab["Date_dt"].min()).date() if not fab["Date_dt"].isna().all() else date.today() - timedelta(days=30)
    max_f = pd.to_datetime(fab["Date_dt"].max()).date() if not fab["Date_dt"].isna().all() else date.today()

    f1, f2, f3 = st.columns(3)
    f_from = f1.date_input("Du", value=min_f, key="fab_from")
    f_to   = f2.date_input("Au", value=max_f, key="fab_to")
    prod_pick = f3.multiselect("Produit", options=["GMQ ONE","GMQ LIVE"], default=["GMQ ONE","GMQ LIVE"], key="fab_prod")

    f4, f5 = st.columns(2)
    status_opts = ["(Tous)"] + sorted(fab["Status"].dropna().astype(str).unique().tolist()) if "Status" in fab.columns else ["(Tous)"]
    status_pick = f4.selectbox("Statut", status_opts, index=0, key="fab_status")
    client_filter = f5.text_input("Client contient", "", key="fab_client_q")

    fab_view = fab.drop(columns=["Date_dt"]).copy()
    mask_fd = (fab["Date_dt"].dt.date >= f_from) & (fab["Date_dt"].dt.date <= f_to)
    fab_view = fab_view[mask_fd]
    fab_view = fab_view[fab_view["Product"].isin(prod_pick)]
    if status_pick != "(Tous)" and "Status" in fab_view.columns:
        fab_view = fab_view[fab_view["Status"].astype(str) == status_pick]
    if client_filter.strip() and "Client" in fab_view.columns:
        fab_view = fab_view[fab_view["Client"].astype(str).str.contains(client_filter, case=False, na=False)]

    cols = [c for c in ["MO_ID","Date","DueDate","Product","Qty","Status","Ref","Responsable","Client"] if c in fab_view.columns]
    st.dataframe(fab_view[cols].sort_values(["DueDate","Date"], ascending=[True, False]), use_container_width=True)

# ---- STOCK (ajout + édition rapide + filtres + export)
with tab_stock:
    st.header("Stock")

    # ----- Ajout d’article générique
    st.subheader("Ajouter un article au stock")
    with st.form("stock_add"):
        c1, c2, c3 = st.columns(3)
        sku_new = c1.text_input("SKU *", "", key="stock_sku_new")
        name_new = c2.text_input("Nom *", "", key="stock_name_new")
        unit_new = c3.text_input("Unité", value="pcs", key="stock_unit_new")
        c4, c5, c6 = st.columns(3)
        cat_new = c4.text_input("Catégorie", value="Component", key="stock_cat_new")
        rop_new = c5.number_input("ReorderPoint", min_value=0.0, step=1.0, value=0.0, key="stock_rop_new")
        qty_new = c6.number_input("QtyOnHand (initiale)", min_value=0.0, step=1.0, value=0.0, key="stock_qty_new")
        desc_new = st.text_input("Description", "", key="stock_desc_new")
        btn_add_stock = st.form_submit_button("Ajouter")

        if btn_add_stock:
            if not sku_new.strip() or not name_new.strip():
                st.error("SKU et Nom sont obligatoires.")
            elif sku_new in dfs["Stock"]["SKU"].astype(str).tolist():
                st.error("Ce SKU existe déjà.")
            else:
                new_row = {
                    "SKU": sku_new, "Name": name_new, "Unit": unit_new, "Category": cat_new,
                    "ReorderPoint": float(rop_new), "QtyOnHand": float(qty_new), "Description": desc_new
                }
                dfs["Stock"] = pd.concat([dfs["Stock"], pd.DataFrame([new_row])], ignore_index=True)
                # Met à jour l’index pour cohérence immédiate
                stock_index = build_stock_index(dfs["Stock"])
                if autosave:
                    write_excel_to_path_atomic(dfs, excel_path)
                st.success(f"Article {sku_new} ajouté.")
                st.toast("Article ajouté")

    st.divider()

    # ----- Édition rapide (entière)
    st.subheader("Édition rapide du stock")
    edited_stock = st.data_editor(
        dfs["Stock"],
        use_container_width=True,
        key="stock_editor",
        num_rows="dynamic",
        column_config={
            "ReorderPoint": st.column_config.NumberColumn(step=1.0, min_value=0.0),
            "QtyOnHand": st.column_config.NumberColumn(step=1.0, min_value=0.0)
        }
    )

    if st.button("💾 Enregistrer modifications du stock", key="stock_save_btn"):
        # On impose l’ordre/colonnes du schéma pour éviter toute dérive
        cols = list(SCHEMA["Stock"].keys())
        missing = [c for c in cols if c not in edited_stock.columns]
        for c in missing:
            edited_stock[c] = np.nan if c in ["ReorderPoint","QtyOnHand"] else None
        edited_stock = edited_stock[cols].copy()
        # Types
        edited_stock["ReorderPoint"] = pd.to_numeric(edited_stock["ReorderPoint"], errors="coerce").astype("float64")
        edited_stock["QtyOnHand"] = pd.to_numeric(edited_stock["QtyOnHand"], errors="coerce").astype("float64")
        for c in ["SKU","Name","Unit","Category","Description"]:
            edited_stock[c] = edited_stock[c].astype("object")

        dfs["Stock"] = edited_stock.reset_index(drop=True)
        stock_index = build_stock_index(dfs["Stock"])
        if autosave:
            write_excel_to_path_atomic(dfs, excel_path)
        st.success("Modifications du stock enregistrées.")
        st.toast("Stock enregistré")

    st.divider()

    # ----- Filtres + export pour visualisation
    st.subheader("Recherche & Export du stock")
    s1, s2, s3 = st.columns([1,1,2])
    # Catégories (nettoyage NaN)
    categories = sorted([c for c in dfs["Stock"]["Category"].dropna().astype(str).unique().tolist()])
    cat_pick = s1.multiselect("Catégories", options=categories, default=[], key="stock_cat_filter")
    only_low = s2.checkbox("Sous seuil uniquement", value=False, key="stock_low_filter")
    q_stock = s3.text_input("Recherche (SKU / Nom / Description)", "", key="stock_search")

    stock_view = dfs["Stock"].copy()
    if cat_pick:
        stock_view = stock_view[stock_view["Category"].astype(str).isin(cat_pick)]
    if only_low and "ReorderPoint" in stock_view.columns:
        stock_view = stock_view[ stock_view["QtyOnHand"] < stock_view["ReorderPoint"] ]
    if q_stock.strip():
        mask = (
            stock_view["SKU"].astype(str).str.contains(q_stock, case=False, na=False) |
            stock_view["Name"].astype(str).str.contains(q_stock, case=False, na=False) |
            stock_view["Description"].astype(str).str.contains(q_stock, case=False, na=False)
        )
        stock_view = stock_view[mask]

    st.dataframe(stock_view, use_container_width=True)

    st.download_button(
        "⬇️ Télécharger le stock filtré (CSV)",
        data=to_csv_bytes(stock_view),
        file_name=f"stock_filtre_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="stock_export_btn"
    )

# ---- COMPOSANTS : recherche + ajout (reste utile pour vue ciblée)
with tab_compos:
    st.subheader("Recherche de composants")
    q = st.text_input("Recherche (SKU / Nom / Description)", "", key="comp_q")
    comp_df = dfs["Stock"][ dfs["Stock"]["Category"].astype(str).str.lower().eq("component") ].copy()
    if q.strip():
        mask = (
            comp_df["SKU"].astype(str).str.contains(q, case=False, na=False) |
            comp_df["Name"].astype(str).str.contains(q, case=False, na=False) |
            comp_df["Description"].astype(str).str.contains(q, case=False, na=False)
        )
        comp_df = comp_df[mask]
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("### Ajouter un nouveau composant")
    with st.form("add_component"):
        c1, c2, c3 = st.columns(3)
        sku_new = c1.text_input("SKU *", "", key="comp_sku_new")
        name_new = c2.text_input("Nom *", "", key="comp_name_new")
        unit_new = c3.text_input("Unité", value="pcs", key="comp_unit_new")
        c4, c5, c6 = st.columns(3)
        cat_new = c4.text_input("Catégorie", value="Component", key="comp_cat_new")
        rop_new = c5.number_input("ReorderPoint", min_value=0.0, step=1.0, value=0.0, key="comp_rop_new")
        qty_new = c6.number_input("QtyOnHand (initiale)", min_value=0.0, step=1.0, value=0.0, key="comp_qty_new")
        desc_new = st.text_input("Description", "", key="comp_desc_new")
        btn_add = st.form_submit_button("Ajouter")

        if btn_add:
            if not sku_new.strip() or not name_new.strip():
                st.error("SKU et Nom sont obligatoires.")
            elif sku_new in dfs["Stock"]["SKU"].astype(str).tolist():
                st.error("Ce SKU existe déjà.")
            else:
                new_row = {
                    "SKU": sku_new, "Name": name_new, "Unit": unit_new, "Category": cat_new,
                    "ReorderPoint": float(rop_new), "QtyOnHand": float(qty_new), "Description": desc_new
                }
                dfs["Stock"] = pd.concat([dfs["Stock"], pd.DataFrame([new_row])], ignore_index=True)
                stock_index = build_stock_index(dfs["Stock"])
                if autosave:
                    write_excel_to_path_atomic(dfs, excel_path)
                st.success(f"Composant {sku_new} ajouté.")

# ---- INVENTAIRE : saisie comptage + ajustements
with tab_invent:
    st.subheader("Inventaire (comptage & écarts)")
    resp_list = dfs["Responsables"]["Responsable"].dropna().astype(str).tolist() or ["Aymane","Joslain","Lise","Robin"]
    responsable_inv = st.selectbox("Responsable inventaire", resp_list, index=0, key="inv_resp")
    ref_inv = st.text_input("Référence d'inventaire", value=f"INV-{datetime.now():%Y%m%d}", key="inv_ref")

    st.markdown("**Saisir le comptage** (SKU + quantité comptée) – lignes dynamiques")
    template = pd.DataFrame({"SKU": [], "Compté": []})
    edited = st.data_editor(
        template,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "SKU": st.column_config.SelectboxColumn(options=dfs["Stock"]["SKU"].astype(str).tolist()),
            "Compté": st.column_config.NumberColumn(min_value=0.0, step=1.0)
        },
        key="inv_editor"
    )

    c1, c2 = st.columns(2)
    calc = c1.button("Calculer les écarts", key="inv_calc")
    valider = c2.button("Valider ajustements", key="inv_valid")

    def _compute_diffs(ed: pd.DataFrame) -> pd.DataFrame:
        if ed is None or ed.empty:
            return pd.DataFrame(columns=["SKU","Systeme","Compté","Ecart","Sens"])
        sm = dfs["Stock"][["SKU","QtyOnHand"]].copy()
        sm["SKU"] = sm["SKU"].astype(str)
        ed2 = ed.copy()
        ed2["SKU"] = ed2["SKU"].astype(str)
        merged = ed2.merge(sm, on="SKU", how="left").fillna({"QtyOnHand": 0.0})
        merged.rename(columns={"QtyOnHand":"Systeme"}, inplace=True)
        merged["Compté"] = pd.to_numeric(merged["Compté"], errors="coerce").fillna(0.0)
        merged["Ecart"] = merged["Compté"] - merged["Systeme"]
        merged["Sens"] = np.where(merged["Ecart"]>=0, "IN", "OUT")
        return merged[["SKU","Systeme","Compté","Ecart","Sens"]]

    if calc:
        diffs = _compute_diffs(edited)
        st.markdown("#### Écarts calculés")
        st.dataframe(diffs, use_container_width=True)

    if valider:
        diffs = _compute_diffs(edited)
        if diffs.empty:
            st.info("Aucune ligne à ajuster.")
        else:
            try:
                for r in diffs.itertuples(index=False):
                    sku = r.SKU
                    ecart = float(r.Ecart)
                    if ecart == 0:
                        continue
                    move_type = "IN" if ecart > 0 else "OUT"
                    qty = abs(ecart)
                    dfs["Stock"], _ = apply_movement(dfs["Stock"], build_stock_index(dfs["Stock"]), sku, qty, move_type)
                    dfs["Mouvements"] = record_move(dfs["Mouvements"], sku, move_type, qty, ref_inv, "INVENTAIRE", mo_id=None, responsable=responsable_inv)
                if autosave:
                    write_excel_to_path_atomic(dfs, excel_path)
                st.success("Ajustements d'inventaire enregistrés")
            except PermissionError as e:
                st.error(str(e))

# ---- CLIENTS : ajout + suppression
with tab_clients:
    st.subheader("Clients")

    st.markdown("### Ajouter un client")
    with st.form("add_client"):
        c1, c2 = st.columns(2)
        cname = c1.text_input("Nom du client *", "", key="cli_name")
        ctype = c2.selectbox("Type", ["Régulier","Passage"], key="cli_type")
        c3, c4, c5 = st.columns(3)
        cphone = c3.text_input("Téléphone", "", key="cli_phone")
        cemail = c4.text_input("Email", "", key="cli_email")
        cnotes = c5.text_input("Notes", "", key="cli_notes")
        btn_cli = st.form_submit_button("Ajouter")

        if btn_cli:
            if not cname.strip():
                st.error("Le nom du client est obligatoire.")
            else:
                cid = "C-" + uuid.uuid4().hex[:8].upper()
                row = {
                    "ClientID": cid,
                    "ClientName": cname.strip(),
                    "Type": ctype,
                    "Phone": cphone.strip() or None,
                    "Email": cemail.strip() or None,
                    "Notes": cnotes.strip() or None
                }
                dfs["Clients"] = pd.concat([dfs["Clients"], pd.DataFrame([row])], ignore_index=True)
                if autosave:
                    write_excel_to_path_atomic(dfs, excel_path)
                st.success(f"Client « {cname.strip()} » ajouté.")
                st.toast("Client ajouté")

    st.markdown("### Rechercher / Supprimer des clients")
    cq = st.text_input("Recherche client (nom, téléphone, email)", "", key="clients_search")
    cl = dfs["Clients"].copy()
    if cq.strip():
        m = (
            cl["ClientName"].astype(str).str.contains(cq, case=False, na=False) |
            cl["Phone"].astype(str).str.contains(cq, case=False, na=False) |
            cl["Email"].astype(str).str.contains(cq, case=False, na=False)
        )
        cl = cl[m]

    st.dataframe(cl, use_container_width=True)
    del_ids = st.multiselect(
        "Sélectionne les clients à supprimer",
        options=cl["ClientID"].astype(str).tolist(),
        format_func=lambda cid: f"{cid} – {cl.loc[cl['ClientID']==cid,'ClientName'].values[0] if (cl['ClientID']==cid).any() else cid}",
        key="cli_del_ids"
    )
    if st.button("🗑️ Supprimer la sélection", key="cli_del_btn"):
        if not del_ids:
            st.info("Aucun client sélectionné.")
        else:
            used = dfs["Fabrications"]["Client"].dropna().astype(str)
            names_to_del = dfs["Clients"].loc[dfs["Clients"]["ClientID"].isin(del_ids),"ClientName"].astype(str).tolist()
            referenced = [n for n in names_to_del if n in set(used)]
            if referenced:
                st.warning("Attention : des ordres de fabrication référencent ces clients : " + ", ".join(referenced))
            before = len(dfs["Clients"])
            dfs["Clients"] = dfs["Clients"][~dfs["Clients"]["ClientID"].isin(del_ids)].reset_index(drop=True)
            if autosave:
                write_excel_to_path_atomic(dfs, excel_path)
            st.success(f"Suppression effectuée ({before - len(dfs['Clients'])} client(s)).")

# ---- EXPORT CSV (global, pour convenance)
with tab_export:
    st.subheader("Exports CSV")

    # --- Export STOCK
    st.markdown("### Export Stock")
    col_s1, col_s2 = st.columns(2)
    cats = ["(Toutes)"] + sorted([c for c in dfs["Stock"]["Category"].dropna().astype(str).unique().tolist()])
    cat_pick = col_s1.selectbox("Catégorie", cats, index=0, key="exp_stock_cat")
    only_low = col_s2.checkbox("Seulement sous seuil", value=False, key="exp_stock_low")

    stock_exp = dfs["Stock"].copy()
    if cat_pick != "(Toutes)":
        stock_exp = stock_exp[ stock_exp["Category"].astype(str) == cat_pick ]
    if only_low and "ReorderPoint" in stock_exp.columns:
        stock_exp = stock_exp[ stock_exp["QtyOnHand"] < stock_exp["ReorderPoint"] ]
    st.download_button(
        "⬇️ Télécharger Stock filtré (CSV)",
        data=to_csv_bytes(stock_exp),
        file_name=f"stock_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="exp_stock_btn"
    )

    st.divider()

    # --- Export MOUVEMENTS
    st.markdown("### Export Mouvements")
    mv = dfs["Mouvements"].copy()
    mv["Date_dt"] = pd.to_datetime(mv["Date"], errors="coerce")
    min_d = pd.to_datetime(mv["Date_dt"].min()).date() if not mv["Date_dt"].isna().all() else date.today() - timedelta(days=30)
    max_d = pd.to_datetime(mv["Date_dt"].max()).date() if not mv["Date_dt"].isna().all() else date.today()

    c1, c2, c3 = st.columns(3)
    d_from = c1.date_input("Du", value=min_d, key="exp_mv_from")
    d_to   = c2.date_input("Au", value=max_d, key="exp_mv_to")
    types = c3.multiselect("Type", options=["IN","OUT"], default=["IN","OUT"], key="exp_mv_types")

    c4, c5 = st.columns(2)
    sku_filter = c4.text_input("Filtre SKU (contient)", "", key="exp_mv_sku")
    resp_opts = ["(Tous)"] + sorted(mv["Responsable"].dropna().astype(str).unique().tolist())
    resp_pick = c5.selectbox("Responsable", resp_opts, index=0, key="exp_mv_resp")

    mv_exp = mv.drop(columns=["Date_dt"]).copy()
    mask_date = (mv["Date_dt"].dt.date >= d_from) & (mv["Date_dt"].dt.date <= d_to)
    mv_exp = mv_exp[mask_date]
    mv_exp = mv_exp[mv["Type"].isin(types)]
    if sku_filter.strip():
        mv_exp = mv_exp[mv_exp["SKU"].astype(str).str.contains(sku_filter, case=False, na=False)]
    if resp_pick != "(Tous)":
        mv_exp = mv_exp[mv_exp["Responsable"].astype(str) == resp_pick]

    st.download_button(
        "⬇️ Télécharger Mouvements filtrés (CSV)",
        data=to_csv_bytes(mv_exp),
        file_name=f"mouvements_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="exp_mv_btn"
    )

    st.divider()

    # --- Export FABRICATIONS
    st.markdown("### Export Fabrications")
    fab = dfs["Fabrications"].copy()
    fab["Date_dt"] = pd.to_datetime(fab["Date"], errors="coerce")
    min_f = pd.to_datetime(fab["Date_dt"].min()).date() if not fab["Date_dt"].isna().all() else date.today() - timedelta(days=30)
    max_f = pd.to_datetime(fab["Date_dt"].max()).date() if not fab["Date_dt"].isna().all() else date.today()

    f1, f2, f3 = st.columns(3)
    f_from = f1.date_input("Du", value=min_f, key="exp_fab_from")
    f_to   = f2.date_input("Au", value=max_f, key="exp_fab_to")
    prod_pick = f3.multiselect("Produit", options=["GMQ ONE","GMQ LIVE"], default=["GMQ ONE","GMQ LIVE"], key="exp_fab_prod")

    f4, f5 = st.columns(2)
    status_opts = ["(Tous)"] + sorted(fab["Status"].dropna().astype(str).unique().tolist()) if "Status" in fab.columns else ["(Tous)"]
    status_pick = f4.selectbox("Statut", status_opts, index=0, key="exp_fab_status")
    client_filter = f5.text_input("Client contient", "", key="exp_fab_client")

    fab_exp = fab.drop(columns=["Date_dt"]).copy()
    mask_fd = (fab["Date_dt"].dt.date >= f_from) & (fab["Date_dt"].dt.date <= f_to)
    fab_exp = fab_exp[mask_fd]
    fab_exp = fab_exp[fab_exp["Product"].isin(prod_pick)]
    if status_pick != "(Tous)" and "Status" in fab_exp.columns:
        fab_exp = fab_exp[fab_exp["Status"].astype(str) == status_pick]
    if client_filter.strip() and "Client" in fab_exp.columns:
        fab_exp = fab_exp[fab_exp["Client"].astype(str).str.contains(client_filter, case=False, na=False)]

    st.download_button(
        "⬇️ Télécharger Fabrications filtrées (CSV)",
        data=to_csv_bytes(fab_exp),
        file_name=f"fabrications_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        key="exp_fab_btn"
    )
