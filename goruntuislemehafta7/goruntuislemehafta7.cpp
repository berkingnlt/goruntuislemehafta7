#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

using namespace cv;
using namespace std;

// Onceki maddelerdeki sinus sinyali.
vector<double> sinusOlustur(
    int N, double genlik, double fazDerece,
    double frekans, double fs)
{
    vector<double> veri(N);
    double faz = fazDerece * CV_PI / 180.0;

    for (int n = 0; n < N; n++)
    {
        veri[n] = genlik * sin(
            2.0 * CV_PI * frekans * n / fs + faz
        );
    }

    return veri;
}

// Onceki maddelerdeki ayni gurultu.
vector<double> gurultuEkle(
    const vector<double>& veri,
    double oran)
{
    if (oran < 0.0 || oran > 1.0)
        throw invalid_argument("Gurultu orani [0,1] olmali.");

    double tepe = 0.0;

    for (double deger : veri)
        if (abs(deger) > tepe)
            tepe = abs(deger);

    mt19937 uretec(42);
    uniform_real_distribution<double> dagilim(-1.0, 1.0);

    vector<double> sonuc = veri;

    for (size_t n = 0; n < veri.size(); n++)
        sonuc[n] += dagilim(uretec) * tepe * oran;

    return sonuc;
}

// MANUEL DFT:
// X[k] = toplam_n x[n] * exp(-j*2*pi*k*n/N)
//
// Harmonik vektoru karmasik sayilar olarak dondurur.
vector<complex<double>> kendiDFT(
    const vector<double>& veri)
{
    size_t boyut = veri.size();

    if (boyut < 2 || (boyut & (boyut - 1)) != 0)
    {
        throw invalid_argument(
            "Vektor uzunlugu en az 2 ve 2'nin kuvveti olmali."
        );
    }

    int N = static_cast<int>(boyut);
    vector<complex<double>> harmonikler(N);

    for (int k = 0; k < N; k++)
    {
        double gercek = 0.0;
        double sanal = 0.0;

        for (int n = 0; n < N; n++)
        {
            double aci =
                2.0 * CV_PI * k * n / N;

            gercek += veri[n] * cos(aci);
            sanal -= veri[n] * sin(aci);
        }

        harmonikler[k] = complex<double>(gercek, sanal);
    }

    return harmonikler;
}

// Gercek sinyal icin tek tarafli genlik spektrumu.
// DC ve Nyquist disindaki frekanslar ikiyle carpilir.
vector<double> genlikSpektrumu(
    const vector<complex<double>>& harmonikler)
{
    int N = static_cast<int>(harmonikler.size());
    vector<double> genlikler(N / 2 + 1);

    for (int k = 0; k <= N / 2; k++)
    {
        double deger = abs(harmonikler[k]) / N;

        if (k != 0 && k != N / 2)
            deger *= 2.0;

        genlikler[k] = deger;
    }

    return genlikler;
}

// DC haric en buyuk genlige sahip frekans kutusu.
int baskinFrekansIndeksi(const vector<double>& spektrum)
{
    int indeks = 1;

    for (int k = 2; k < static_cast<int>(spektrum.size()); k++)
    {
        if (spektrum[k] > spektrum[indeks])
            indeks = k;
    }

    return indeks;
}

// Spektrumu dikey cubuklarla ciz.
// Goruntulemede ilk 40 Hz'e yakinlastirilir.
Mat spektrumCiz(
    const vector<double>& spektrum,
    double fs,
    int N,
    double gosterilecekMaxHz,
    double ySiniri,
    const string& baslik,
    Scalar renk)
{
    const int W = 1100;
    const int H = 500;

    const int sol = 85;
    const int ust = 65;
    const int grafikW = 980;
    const int grafikH = 350;

    Mat grafik(H, W, CV_8UC3, Scalar(255, 255, 255));

    for (int i = 0; i <= 4; i++)
    {
        int y = ust + cvRound(i * grafikH / 4.0);
        double deger = ySiniri * (1.0 - i / 4.0);

        line(
            grafik, Point(sol, y),
            Point(sol + grafikW, y),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.2f", deger),
            Point(8, y + 5), FONT_HERSHEY_SIMPLEX,
            0.45, Scalar(0, 0, 0), 1
        );
    }

    for (int i = 0; i <= 8; i++)
    {
        int x = sol + cvRound(i * grafikW / 8.0);
        double frekans = i * gosterilecekMaxHz / 8.0;

        line(
            grafik, Point(x, ust),
            Point(x, ust + grafikH),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.1f", frekans),
            Point(x - 15, ust + grafikH + 25),
            FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(0, 0, 0), 1
        );
    }

    rectangle(
        grafik, Point(sol, ust),
        Point(sol + grafikW, ust + grafikH),
        Scalar(0, 0, 0)
    );

    for (int k = 0; k < static_cast<int>(spektrum.size()); k++)
    {
        double frekans = static_cast<double>(k) * fs / N;

        if (frekans > gosterilecekMaxHz)
            break;

        int x = sol + cvRound(
            frekans / gosterilecekMaxHz * grafikW
        );

        int y = ust + grafikH - cvRound(
            spektrum[k] / ySiniri * grafikH
        );

        line(
            grafik,
            Point(x, ust + grafikH),
            Point(x, y),
            renk, 2, LINE_AA
        );

        circle(grafik, Point(x, y), 3, renk, FILLED);
    }

    putText(
        grafik, baslik, Point(sol, 32),
        FONT_HERSHEY_SIMPLEX, 0.65,
        Scalar(0, 0, 0), 2
    );

    putText(
        grafik, "Genlik", Point(8, 35),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    putText(
        grafik, "Frekans (Hz)", Point(500, 480),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    return grafik;
}

int main()
{
    try
    {
        const int N = 512;
        const double fs = 512.0;

        vector<double> temiz =
            sinusOlustur(N, 2.0, 30.0, 5.0, fs);

        vector<double> gurultulu =
            gurultuEkle(temiz, 0.30);

        // Iki vektoru ayni DFT fonksiyonuna ayri gonder.
        vector<complex<double>> temizHarmonikler =
            kendiDFT(temiz);

        vector<complex<double>> gurultuluHarmonikler =
            kendiDFT(gurultulu);

        vector<double> temizSpektrum =
            genlikSpektrumu(temizHarmonikler);

        vector<double> gurultuluSpektrum =
            genlikSpektrumu(gurultuluHarmonikler);

        int temizTepe = baskinFrekansIndeksi(temizSpektrum);
        int gurultuluTepe = baskinFrekansIndeksi(gurultuluSpektrum);

        cout << "Frekans cozunurlugu: " << fs / N
            << " Hz" << endl;

        cout << "\nTemiz sinyalin baskin frekansi: "
            << temizTepe * fs / N << " Hz" << endl;

        cout << "Temiz sinyalin tepe genligi: "
            << temizSpektrum[temizTepe] << endl;

        cout << "\nGurultulu sinyalin baskin frekansi: "
            << gurultuluTepe * fs / N << " Hz" << endl;

        cout << "Gurultulu sinyalin tepe genligi: "
            << gurultuluSpektrum[gurultuluTepe] << endl;

        // Tum N karmasik DFT katsayisini kaydet.
        ofstream harmonikCSV("gorev4_harmonikler.csv");

        if (!harmonikCSV)
            throw runtime_error("Harmonik CSV acilamadi.");

        harmonikCSV
            << "k,isaretli_frekans_hz,temiz_gercek,temiz_sanal,"
            << "gurultulu_gercek,gurultulu_sanal\n";

        for (int k = 0; k < N; k++)
        {
            // Ikinci yari negatif frekanslara karsilik gelir.
            double frekans = (k <= N / 2)
                ? k * fs / N
                : (k - N) * fs / N;

            harmonikCSV
                << k << "," << frekans << ","
                << temizHarmonikler[k].real() << ","
                << temizHarmonikler[k].imag() << ","
                << gurultuluHarmonikler[k].real() << ","
                << gurultuluHarmonikler[k].imag() << "\n";
        }

        harmonikCSV.close();

        // Tek tarafli spektrumun TAMAMINI kaydet: 0-256 Hz.
        ofstream spektrumCSV("gorev4_spektrum.csv");

        if (!spektrumCSV)
            throw runtime_error("Spektrum CSV acilamadi.");

        spektrumCSV << "frekans_hz,temiz_genlik,gurultulu_genlik\n";

        double enBuyukGenlik = 0.0;

        for (int k = 0; k <= N / 2; k++)
        {
            spektrumCSV
                << k * fs / N << ","
                << temizSpektrum[k] << ","
                << gurultuluSpektrum[k] << "\n";

            if (temizSpektrum[k] > enBuyukGenlik)
                enBuyukGenlik = temizSpektrum[k];

            if (gurultuluSpektrum[k] > enBuyukGenlik)
                enBuyukGenlik = gurultuluSpektrum[k];
        }

        spektrumCSV.close();

        double ySiniri = enBuyukGenlik * 1.15;

        if (ySiniri <= 0)
            ySiniri = 1.0;

        // Ana tepeyi ve gurultuyu rahat gormek icin 0-40 Hz.
        Mat temizGrafik = spektrumCiz(
            temizSpektrum, fs, N, 40.0, ySiniri,
            "Temiz Sinyal - Manuel DFT (0-40 Hz)",
            Scalar(220, 70, 0)
        );

        Mat gurultuluGrafik = spektrumCiz(
            gurultuluSpektrum, fs, N, 40.0, ySiniri,
            "Gurultulu Sinyal - Manuel DFT (0-40 Hz)",
            Scalar(0, 0, 220)
        );

        imwrite("gorev4_temiz_spektrum.png", temizGrafik);
        imwrite("gorev4_gurultulu_spektrum.png", gurultuluGrafik);

        ofstream rapor("gorev4_yorum.txt");

        if (!rapor)
            throw runtime_error("Rapor dosyasi acilamadi.");

        rapor << "Manuel DFT, N=" << N << ", fs=" << fs << " Hz\n";
        rapor << "Frekans cozunurlugu: " << fs / N << " Hz\n";

        rapor << "Temiz baskin frekans: "
            << temizTepe * fs / N << " Hz\n";

        rapor << "Temiz tepe genligi: "
            << temizSpektrum[temizTepe] << "\n";

        rapor << "Gurultulu baskin frekans: "
            << gurultuluTepe * fs / N << " Hz\n";

        rapor << "Gurultulu tepe genligi: "
            << gurultuluSpektrum[gurultuluTepe] << "\n\n";

        rapor
            << "Temiz sinyal 5 Hz sinus oldugu icin spektrumda "
            << "5 Hz'de yaklasik 2 genlikli tepe beklenir.\n"
            << "Sinyal kayit boyunca tam 5 periyot tamamlar; "
            << "bu nedenle frekans kutularina uyumludur.\n"
            << "Rastgele gurultu diger frekans kutularinda da "
            << "bilesenler olusturur ve ana tepeyi degistirebilir.\n"
            << "Grafikler 0-40 Hz'i, spektrum CSV dosyasi ise "
            << "0-256 Hz araliginin tamamini gosterir.\n";

        rapor.close();

        imshow("Temiz Sinyal - DFT", temizGrafik);
        imshow("Gurultulu Sinyal - DFT", gurultuluGrafik);

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}