#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace cv;
using namespace std;

// MANUEL 2 BOYUTLU DFT.
//
// F(v,u) = toplam_y toplam_x:
// f(y,x) * exp(-j*2*pi*(u*x/W + v*y/H))
//
// Donusum ayrilabilir:
// Once satir DFT'leri, sonra sutun DFT'leri.
//
// Donen vektorun indeksi:
// harmonikler[v * genislik + u]
vector<complex<double>> kendiDFT2D(const Mat& gri)
{
    if (gri.empty() || gri.type() != CV_8UC1)
    {
        throw invalid_argument(
            "Girdi 8 bit, tek kanalli gri goruntu olmali."
        );
    }

    int H = gri.rows;
    int W = gri.cols;

    size_t boyut = static_cast<size_t>(H) * W;

    vector<complex<double>> satirSonuclari(boyut);
    vector<complex<double>> harmonikler(boyut);

    // 1. Her satirin DFT'sini hesapla.
    for (int y = 0; y < H; y++)
    {
        for (int u = 0; u < W; u++)
        {
            complex<double> toplam(0.0, 0.0);

            for (int x = 0; x < W; x++)
            {
                // Piksel degerini [0,1] araligina getir.
                double piksel =
                    gri.at<uchar>(y, x) / 255.0;

                double aci =
                    -2.0 * CV_PI * u * x / W;

                complex<double> katsayi(
                    cos(aci),
                    sin(aci)
                );

                toplam += piksel * katsayi;
            }

            satirSonuclari[
                static_cast<size_t>(y) * W + u
            ] = toplam;
        }
    }

    // 2. Satir sonuclarinin her sutununa DFT uygula.
    for (int v = 0; v < H; v++)
    {
        for (int u = 0; u < W; u++)
        {
            complex<double> toplam(0.0, 0.0);

            for (int y = 0; y < H; y++)
            {
                double aci =
                    -2.0 * CV_PI * v * y / H;

                complex<double> katsayi(
                    cos(aci),
                    sin(aci)
                );

                toplam +=
                    satirSonuclari[
                        static_cast<size_t>(y) * W + u
                    ] * katsayi;
            }

            harmonikler[
                static_cast<size_t>(v) * W + u
            ] = toplam;
        }
    }

    return harmonikler;
}

// Harmonik vektorunu goruntulenebilir genlik spektrumuna cevir.
// Bu islem hesaplanmis harmonik katsayilarini degistirmez.
Mat spektrumGoruntusu(
    const vector<complex<double>>& harmonikler,
    int H,
    int W)
{
    if (harmonikler.size() != static_cast<size_t>(H) * W)
        throw invalid_argument("Harmonik boyutu uyusmuyor.");

    Mat merkezlenmis(H, W, CV_64FC1);

    for (int v = 0; v < H; v++)
    {
        for (int u = 0; u < W; u++)
        {
            double genlik = abs(
                harmonikler[static_cast<size_t>(v) * W + u]
            );

            // Genlik farklarini daha rahat gormek icin log.
            double logGenlik = std::log1p(genlik);

            // Dusuk frekanslari goruntunun merkezine tasi.
            int yeniY = (v + H / 2) % H;
            int yeniX = (u + W / 2) % W;

            merkezlenmis.at<double>(yeniY, yeniX) = logGenlik;
        }
    }

    Mat sonuc;

    normalize(
        merkezlenmis,
        sonuc,
        0,
        255,
        NORM_MINMAX,
        CV_8U
    );

    return sonuc;
}

int main()
{
    try
    {
        Mat orijinal = imread("resim7.jpg", IMREAD_COLOR);

        if (orijinal.empty())
        {
            cout << "resim7.jpg bulunamadi!" << endl;
            return -1;
        }

        // 1. Griye donustur.
        Mat gri;
        cvtColor(orijinal, gri, COLOR_BGR2GRAY);

        // Manuel DFT maliyetini azaltmak icin.
        // DFT analizi bu yeniden boyutlandirilmis goruntuye aittir.
        Mat analizGoruntusu;
        resize(
            gri,
            analizGoruntusu,
            Size(128, 128),
            0,
            0,
            INTER_AREA
        );

        cout << "Analiz boyutu: "
            << analizGoruntusu.cols << " x "
            << analizGoruntusu.rows << endl;

        cout << "Manuel 2B DFT hesaplaniyor..." << endl;

        // 2. Kendi fonksiyonumuzdan harmonik vektorunu al.
        vector<complex<double>> harmonikler =
            kendiDFT2D(analizGoruntusu);

        int H = analizGoruntusu.rows;
        int W = analizGoruntusu.cols;

        cout << "DFT tamamlandi." << endl;
        cout << "Harmonik vektor uzunlugu: "
            << harmonikler.size() << endl;

        // 3. Tum karmasik katsayilari kaydet.
        ofstream csv("gorev7_harmonikler.csv");

        if (!csv)
            throw runtime_error("CSV dosyasi olusturulamadi.");

        csv << setprecision(15);

        csv << "v,u,gercek,sanal,genlik\n";

        for (int v = 0; v < H; v++)
        {
            for (int u = 0; u < W; u++)
            {
                const complex<double>& katsayi =
                    harmonikler[
                        static_cast<size_t>(v) * W + u
                    ];

                csv << v << ","
                    << u << ","
                    << katsayi.real() << ","
                    << katsayi.imag() << ","
                    << abs(katsayi) << "\n";
            }
        }

        csv.close();

        // 4. Genlik spektrumunu olustur.
        Mat spektrum =
            spektrumGoruntusu(harmonikler, H, W);

        // 5. DC katsayisini kontrol et.
        // F(0,0), normalize piksel degerlerinin toplamidir.
        double pikselToplami = 0.0;

        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                pikselToplami +=
                    analizGoruntusu.at<uchar>(y, x) / 255.0;
            }
        }

        double dcHatasi = abs(
            harmonikler[0] -
            complex<double>(pikselToplami, 0.0)
        );

        cout << "\nDC katsayisi F(0,0): "
            << harmonikler[0] << endl;

        cout << "DC kontrol hatasi: "
            << dcHatasi << endl;

        // 6. Sonuclari kaydet.
        imwrite("gorev7_gri_128.png", analizGoruntusu);
        imwrite("gorev7_dft_spektrumu.png", spektrum);

        ofstream rapor("gorev7_bilgi.txt");

        if (!rapor)
            throw runtime_error("Rapor olusturulamadi.");

        rapor << "Yontem: Manuel, ayrilabilir 2 boyutlu DFT\n";
        rapor << "Analiz boyutu: " << W << " x " << H << "\n";
        rapor << "Harmonik sayisi: " << harmonikler.size() << "\n";
        rapor << "Piksel olcegi: [0,1]\n";
        rapor << "DC kontrol hatasi: " << dcHatasi << "\n";
        rapor << "CSV'deki katsayilar merkezlenmemis ham DFT'dir.\n";
        rapor << "Spektrum PNG'si logaritmik, merkezlenmis ve "
            << "0-255 araligina olceklenmis genliktir.\n";

        rapor.close();

        // Kucuk resimleri rahat incelemek icin buyut.
        // Bu buyutme yalniz ekranda gostermek icindir.
        Mat griGosterim, spektrumGosterim;

        resize(
            analizGoruntusu,
            griGosterim,
            Size(512, 512),
            0, 0,
            INTER_NEAREST
        );

        resize(
            spektrum,
            spektrumGosterim,
            Size(512, 512),
            0, 0,
            INTER_NEAREST
        );

        imshow("Orijinal", orijinal);
        imshow("DFT Girisi - Gri Goruntu", griGosterim);
        imshow("Manuel 2B DFT - Genlik Spektrumu", spektrumGosterim);

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}