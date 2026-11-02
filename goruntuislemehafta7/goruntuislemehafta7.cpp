#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <fstream>
#include <stdexcept>

using namespace cv;
using namespace std;

using Harmonik = vector<complex<double>>;

// --------------------------------------------------
// GURULTU FONKSIYONLARI
// --------------------------------------------------

Mat gaussGurultusu(const Mat& gri, double sigma)
{
    Mat sonuc = gri.clone();

    mt19937 uretec(42);
    normal_distribution<double> dagilim(0.0, sigma);

    for (int y = 0; y < gri.rows; y++)
        for (int x = 0; x < gri.cols; x++)
            sonuc.at<uchar>(y, x) = saturate_cast<uchar>(
                gri.at<uchar>(y, x) + dagilim(uretec)
            );

    return sonuc;
}

Mat saltPepperGurultusu(const Mat& gri, double oran)
{
    Mat sonuc = gri.clone();

    mt19937 uretec(123);
    uniform_real_distribution<double> dagilim(0.0, 1.0);

    for (int y = 0; y < gri.rows; y++)
    {
        for (int x = 0; x < gri.cols; x++)
        {
            double p = dagilim(uretec);

            if (p < oran / 2.0)
                sonuc.at<uchar>(y, x) = 0;
            else if (p < oran)
                sonuc.at<uchar>(y, x) = 255;
        }
    }

    return sonuc;
}

// --------------------------------------------------
// MANUEL DFT / TERS DFT
// --------------------------------------------------

Harmonik donusum2D(
    const Harmonik& girdi, int H, int W, bool ters)
{
    size_t boyut = static_cast<size_t>(H) * W;

    if (H <= 0 || W <= 0 || girdi.size() != boyut)
        throw invalid_argument("Donusum boyutlari gecersiz.");

    double isaret = ters ? 1.0 : -1.0;

    Harmonik satirKatsayi(static_cast<size_t>(W) * W);
    Harmonik sutunKatsayi(static_cast<size_t>(H) * H);

    for (int u = 0; u < W; u++)
    {
        for (int x = 0; x < W; x++)
        {
            double aci = isaret * 2.0 * CV_PI * u * x / W;

            satirKatsayi[static_cast<size_t>(u) * W + x] =
                complex<double>(cos(aci), sin(aci));
        }
    }

    for (int v = 0; v < H; v++)
    {
        for (int y = 0; y < H; y++)
        {
            double aci = isaret * 2.0 * CV_PI * v * y / H;

            sutunKatsayi[static_cast<size_t>(v) * H + y] =
                complex<double>(cos(aci), sin(aci));
        }
    }

    Harmonik ara(boyut);
    Harmonik sonuc(boyut);

    for (int y = 0; y < H; y++)
    {
        for (int u = 0; u < W; u++)
        {
            complex<double> toplam(0.0, 0.0);

            for (int x = 0; x < W; x++)
                toplam +=
                girdi[static_cast<size_t>(y) * W + x] *
                satirKatsayi[static_cast<size_t>(u) * W + x];

            ara[static_cast<size_t>(y) * W + u] = toplam;
        }
    }

    for (int v = 0; v < H; v++)
    {
        for (int u = 0; u < W; u++)
        {
            complex<double> toplam(0.0, 0.0);

            for (int y = 0; y < H; y++)
                toplam +=
                ara[static_cast<size_t>(y) * W + u] *
                sutunKatsayi[static_cast<size_t>(v) * H + y];

            if (ters)
                toplam /= static_cast<double>(boyut);

            sonuc[static_cast<size_t>(v) * W + u] = toplam;
        }
    }

    return sonuc;
}

Harmonik kendiDFT2D(const Mat& resim)
{
    Harmonik veri(resim.total());

    for (int y = 0; y < resim.rows; y++)
        for (int x = 0; x < resim.cols; x++)
            veri[static_cast<size_t>(y) * resim.cols + x] =
            resim.at<double>(y, x);

    return donusum2D(veri, resim.rows, resim.cols, false);
}

Mat kendiTersDFT2D(const Harmonik& F, int H, int W)
{
    Harmonik veri = donusum2D(F, H, W, true);
    Mat sonuc(H, W, CV_64FC1);

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            sonuc.at<double>(y, x) =
            veri[static_cast<size_t>(y) * W + x].real();

    return sonuc;
}

// --------------------------------------------------
// FREKANS FILTRELERI
// Keskin kesim yerine yumusak Gauss gecisleri.
// --------------------------------------------------

double filtreDegeri(
    int tur, double Dkare,
    double D0, double D1, double D2)
{
    double lowPass = exp(-Dkare / (2.0 * D0 * D0));

    double darGauss = exp(-Dkare / (2.0 * D1 * D1));
    double genisGauss = exp(-Dkare / (2.0 * D2 * D2));

    // Iki alcak geciren filtrenin farki.
    double bandPass = genisGauss - darGauss;

    switch (tur)
    {
    case 0: return lowPass;
    case 1: return 1.0 - lowPass;
    case 2: return bandPass;
    case 3: return 1.0 - bandPass;
    }

    throw invalid_argument("Filtre turu gecersiz.");
}

Mat maskeOlustur(
    int H, int W, int tur,
    double D0, double D1, double D2)
{
    Mat maske(H, W, CV_64FC1);

    // DFT matrisi burada merkezlenmemis durumdadir.
    // Sarilmis indislerle sifir frekansa uzakligi bul.
    for (int v = 0; v < H; v++)
    {
        int fv = (v <= H / 2) ? v : v - H;

        for (int u = 0; u < W; u++)
        {
            int fu = (u <= W / 2) ? u : u - W;

            double Dkare =
                static_cast<double>(fu) * fu +
                static_cast<double>(fv) * fv;

            maske.at<double>(v, u) =
                filtreDegeri(tur, Dkare, D0, D1, D2);
        }
    }

    return maske;
}

Harmonik filtreUygula(const Harmonik& F, const Mat& maske)
{
    Harmonik sonuc = F;

    for (int v = 0; v < maske.rows; v++)
        for (int u = 0; u < maske.cols; u++)
            sonuc[static_cast<size_t>(v) * maske.cols + u] *=
            maske.at<double>(v, u);

    return sonuc;
}

// --------------------------------------------------
// KAYDETME VE GORSELLESTIRME
// --------------------------------------------------

Mat merkezle(const Mat& kaynak)
{
    Mat sonuc(kaynak.size(), kaynak.type());

    for (int y = 0; y < kaynak.rows; y++)
        for (int x = 0; x < kaynak.cols; x++)
            sonuc.at<double>(
                (y + kaynak.rows / 2) % kaynak.rows,
                (x + kaynak.cols / 2) % kaynak.cols
            ) = kaynak.at<double>(y, x);

    return sonuc;
}

Mat spektrumGoruntusu(
    const Harmonik& F, int H, int W, double ortakMaksimum)
{
    Mat logGenlik(H, W, CV_64FC1);

    for (int v = 0; v < H; v++)
        for (int u = 0; u < W; u++)
            logGenlik.at<double>(v, u) = std::log1p(
                abs(F[static_cast<size_t>(v) * W + u])
            );

    Mat kaydirilmis = merkezle(logGenlik);
    Mat sonuc;

    kaydirilmis.convertTo(
        sonuc, CV_8U,
        ortakMaksimum > 0 ? 255.0 / ortakMaksimum : 0.0
    );

    return sonuc;
}

// Low-pass / band-reject: normal parlaklik.
// High-pass / band-pass: sifir degeri orta gri.
Mat sonucGoruntusu(const Mat& sonuc, bool isaretli)
{
    Mat goruntu;

    if (isaretli)
        sonuc.convertTo(goruntu, CV_8U, 127.5, 127.5);
    else
        sonuc.convertTo(goruntu, CV_8U, 255.0);

    return goruntu;
}

// Kenar degisimi gostergesi:
// Yatay ve dikey komsularin ortalama mutlak farki.
// Bu deger tek basina gurultu miktari degildir.
double komsuFarki(const Mat& resim)
{
    double toplam = 0.0;
    size_t sayac = 0;

    for (int y = 0; y < resim.rows; y++)
    {
        for (int x = 0; x < resim.cols; x++)
        {
            double p = resim.at<double>(y, x);

            if (x + 1 < resim.cols)
            {
                toplam += abs(p - resim.at<double>(y, x + 1));
                sayac++;
            }

            if (y + 1 < resim.rows)
            {
                toplam += abs(p - resim.at<double>(y + 1, x));
                sayac++;
            }
        }
    }

    return sayac > 0 ? toplam / sayac : 0.0;
}

void istatistikYaz(
    const string& girdi,
    const string& filtre,
    const Mat& sonuc,
    ofstream& csv)
{
    Scalar ortalama, standartSapma;
    meanStdDev(sonuc, ortalama, standartSapma);

    double kenar = komsuFarki(sonuc);

    csv << girdi << "," << filtre << ","
        << ortalama[0] << ","
        << standartSapma[0] << ","
        << kenar << "\n";

    cout << girdi << " / " << filtre
        << " | Ortalama=" << ortalama[0]
        << " | Komsu farki=" << kenar << endl;
}

int main()
{
    try
    {
        Mat kaynak = imread("resim7.jpg", IMREAD_GRAYSCALE);

        if (kaynak.empty())
            throw runtime_error("resim7.jpg bulunamadi!");

        Mat gri;
        resize(kaynak, gri, Size(128, 128),
            0, 0, INTER_AREA);

        vector<Mat> girdiler8 = {
            gri,
            gaussGurultusu(gri, 20.0),
            saltPepperGurultusu(gri, 0.05)
        };

        vector<string> girdiIsimleri = {
            "Orijinal", "Gauss", "SaltPepper"
        };

        vector<string> filtreIsimleri = {
            "LowPass", "HighPass", "BandPass", "BandReject"
        };

        int H = gri.rows;
        int W = gri.cols;

        // Birimler: frekans kutusu.
        // Yumusak gecis olduklari icin keskin sinir degildir.
        const double D0 = 18.0;
        const double D1 = 8.0;
        const double D2 = 30.0;

        vector<Mat> maskeler;

        for (int tur = 0; tur < 4; tur++)
        {
            Mat maske = maskeOlustur(
                H, W, tur, D0, D1, D2
            );

            maskeler.push_back(maske);

            Mat maske8;
            merkezle(maske).convertTo(maske8, CV_8U, 255.0);

            imwrite(
                "gorev10_maske_" + filtreIsimleri[tur] + ".png",
                maske8
            );
        }

        vector<Mat> girdiler;
        vector<Harmonik> harmonikler;

        double ortakSpektrumMaksimumu = 0.0;

        for (int i = 0; i < 3; i++)
        {
            Mat veri;
            girdiler8[i].convertTo(veri, CV_64F, 1.0 / 255.0);

            girdiler.push_back(veri);

            cout << girdiIsimleri[i]
                << " DFT hesaplaniyor..." << endl;

            harmonikler.push_back(kendiDFT2D(veri));

            for (const auto& katsayi : harmonikler.back())
            {
                double deger = std::log1p(abs(katsayi));

                if (deger > ortakSpektrumMaksimumu)
                    ortakSpektrumMaksimumu = deger;
            }
        }

        ofstream csv("gorev10_istatistikler.csv");

        if (!csv)
            throw runtime_error("CSV olusturulamadi.");

        csv.precision(15);

        csv << "girdi,filtre,ortalama,std_sapma,ortalama_komsu_farki\n";

        for (int i = 0; i < 3; i++)
        {
            string onEk = "gorev10_" + girdiIsimleri[i];

            imwrite(onEk + "_girdi.png", girdiler8[i]);

            istatistikYaz(
                girdiIsimleri[i], "Girdi", girdiler[i], csv
            );

            // Ekranda 4 sonucu 2x2 panelde goster.
            Mat panel(572, 512, CV_8UC3, Scalar(255, 255, 255));

            for (int tur = 0; tur < 4; tur++)
            {
                Harmonik filtreliF = filtreUygula(
                    harmonikler[i], maskeler[tur]
                );

                Mat sonuc = kendiTersDFT2D(filtreliF, H, W);

                bool isaretli = (tur == 1 || tur == 2);

                Mat sonuc8 = sonucGoruntusu(sonuc, isaretli);

                Mat spektrum = spektrumGoruntusu(
                    filtreliF, H, W, ortakSpektrumMaksimumu
                );

                string dosyaOnEk =
                    onEk + "_" + filtreIsimleri[tur];

                imwrite(dosyaOnEk + "_sonuc.png", sonuc8);
                imwrite(dosyaOnEk + "_spektrum.png", spektrum);

                // Istatistikler, gosterim icin kaydirilmis
                // 8 bit resimden DEGIL, gercek sonucundan hesaplanir.
                istatistikYaz(
                    girdiIsimleri[i], filtreIsimleri[tur],
                    sonuc, csv
                );

                Mat buyuk, renkli;
                resize(sonuc8, buyuk, Size(256, 256),
                    0, 0, INTER_NEAREST);

                cvtColor(buyuk, renkli, COLOR_GRAY2BGR);

                int px = (tur % 2) * 256;
                int py = (tur / 2) * 286;

                renkli.copyTo(
                    panel(Rect(px, py + 30, 256, 256))
                );

                putText(
                    panel, filtreIsimleri[tur],
                    Point(px + 10, py + 21),
                    FONT_HERSHEY_SIMPLEX, 0.55,
                    Scalar(0, 0, 0), 1
                );
            }

            imwrite(onEk + "_karsilastirma.png", panel);
            imshow(girdiIsimleri[i] + " - Dort Filtre", panel);
        }

        csv.close();

        ofstream rapor("gorev10_yorum.txt");

        if (!rapor)
            throw runtime_error("Rapor olusturulamadi.");

        rapor
            << "MANUEL FREKANS FILTRELERI\n"
            << "Analiz boyutu: 128x128, piksel olcegi [0,1].\n"
            << "Low/High-pass D0=" << D0 << "\n"
            << "Band filtreleri D1=" << D1 << ", D2=" << D2 << "\n\n"
            << "Low-pass: Yuksek frekanslari zayiflatir. "
            << "Gurultuyu azaltabilir; kenarlari yumusatir. "
            << "DC kazanci 1 oldugu icin ortalamayi korur.\n\n"
            << "High-pass: Dusuk frekanslari ve DC'yi bastirir. "
            << "Kenar ve ince detaylari vurgular; gurultuyu da "
            << "belirginlestirebilir. Cikti ortalamasi sifira yakindir.\n\n"
            << "Band-pass: Orta frekanslari gecirir. Genis parlaklik "
            << "degisimleri ve cok ince detaylar zayiflar. "
            << "Gecis bandindaki gurultu kalabilir. DC kazanci 0'dir.\n\n"
            << "Band-reject: Orta frekanslari zayiflatir. "
            << "Bu banttaki detaylar da kaybolabilir. "
            << "DC kazanci 1 oldugu icin ortalamayi korur.\n\n"
            << "High-pass ve band-pass PNG'lerinde sifir degeri "
            << "orta gridir; bu yalniz gosterim donusumudur.\n"
            << "Istatistikler kirpilmis PNG'lerden degil, "
            << "kayan noktali filtre sonuclarindan hesaplanir.\n"
            << "Ortalama komsu farki hem kenarlardan hem gurultuden "
            << "etkilenir; tek basina gurultu olcusu degildir.\n";

        rapor.close();

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}