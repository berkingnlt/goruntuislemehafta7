#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>

using namespace cv;
using namespace std;

// Vektorun uzunlugu 2'nin kuvveti mi?
bool ikininKuvveti(size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

// Parametre olarak verilen vektoru sinus degerleriyle doldurur.
// Genlik: A
// Faz: derece
// Frekans ve ornekleme frekansi: Hz
void sinusOlustur(
    vector<double>& veri,
    double genlik,
    double fazDerece,
    double frekans,
    double orneklemeFrekansi)
{
    if (veri.size() < 2 || !ikininKuvveti(veri.size()))
    {
        throw invalid_argument(
            "Vektor uzunlugu en az 2 ve 2'nin kuvveti olmali."
        );
    }

    if (genlik < 0 || orneklemeFrekansi <= 0 ||
        frekans <= 0 || frekans >= orneklemeFrekansi / 2.0)
    {
        throw invalid_argument(
            "Genlik >= 0 ve 0 < frekans < orneklemeFrekansi/2 olmali."
        );
    }

    double fazRadyan = fazDerece * CV_PI / 180.0;

    for (size_t n = 0; n < veri.size(); n++)
    {
        double zaman =
            static_cast<double>(n) / orneklemeFrekansi;

        // x[n] = A * sin(2*pi*f*t + faz)
        veri[n] = genlik * sin(
            2.0 * CV_PI * frekans * zaman + fazRadyan
        );
    }
}

// Nokta grafigi veya cizgi grafigi olusturur.
Mat grafikCiz(
    const vector<double>& veri,
    double orneklemeFrekansi,
    double genlik,
    bool noktali)
{
    const int genislik = 1100;
    const int yukseklik = 500;

    const int sol = 85;
    const int sag = 30;
    const int ust = 65;
    const int alt = 70;

    int grafikGenisligi = genislik - sol - sag;
    int grafikYuksekligi = yukseklik - ust - alt;

    Mat grafik(
        yukseklik,
        genislik,
        CV_8UC3,
        Scalar(255, 255, 255)
    );

    double ySiniri = genlik > 0 ? genlik * 1.2 : 1.0;

    double sonZaman =
        static_cast<double>(veri.size() - 1) /
        orneklemeFrekansi;

    auto koordinat = [&](size_t n) -> Point
        {
            int x = sol + cvRound(
                static_cast<double>(n) /
                (veri.size() - 1) * grafikGenisligi
            );

            int y = ust + cvRound(
                (ySiniri - veri[n]) /
                (2.0 * ySiniri) * grafikYuksekligi
            );

            return Point(x, y);
        };

    // Yatay izgara ve genlik degerleri.
    for (int i = 0; i <= 4; i++)
    {
        int y = ust + cvRound(
            i * grafikYuksekligi / 4.0
        );

        double deger = ySiniri - i * ySiniri / 2.0;

        line(
            grafik,
            Point(sol, y),
            Point(sol + grafikGenisligi, y),
            Scalar(220, 220, 220),
            1
        );

        putText(
            grafik,
            format("%.2f", deger),
            Point(8, y + 5),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            Scalar(0, 0, 0),
            1
        );
    }

    // Dikey izgara ve zaman degerleri.
    for (int i = 0; i <= 5; i++)
    {
        int x = sol + cvRound(
            i * grafikGenisligi / 5.0
        );

        line(
            grafik,
            Point(x, ust),
            Point(x, ust + grafikYuksekligi),
            Scalar(220, 220, 220),
            1
        );

        putText(
            grafik,
            format("%.2f", i * sonZaman / 5.0),
            Point(x - 18, ust + grafikYuksekligi + 25),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            Scalar(0, 0, 0),
            1
        );
    }

    // Sifir genlik ekseni.
    int sifirY = ust + cvRound(grafikYuksekligi / 2.0);

    line(
        grafik,
        Point(sol, sifirY),
        Point(sol + grafikGenisligi, sifirY),
        Scalar(100, 100, 100),
        1
    );

    rectangle(
        grafik,
        Point(sol, ust),
        Point(sol + grafikGenisligi, ust + grafikYuksekligi),
        Scalar(0, 0, 0),
        1
    );

    // Vektor orneklerini noktalarla goster.
    // Dalga grafiginde ornekleri cizgilerle birlestir.
    for (size_t n = 0; n < veri.size(); n++)
    {
        Point p = koordinat(n);

        if (noktali)
        {
            circle(grafik, p, 2, Scalar(0, 0, 220), FILLED);
        }
        else if (n > 0)
        {
            line(
                grafik,
                koordinat(n - 1),
                p,
                Scalar(220, 70, 0),
                2,
                LINE_AA
            );
        }
    }

    putText(
        grafik,
        noktali
        ? "Sinus Vektoru - Ayrik Ornekler"
        : "Sinus Dalgasi - Ornekler Birlestirilmis",
        Point(sol, 35),
        FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar(0, 0, 0),
        2
    );

    putText(
        grafik, "Genlik",
        Point(8, 35),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    putText(
        grafik, "Zaman (s)",
        Point(genislik / 2 - 40, yukseklik - 15),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    return grafik;
}

int main()
{
    try
    {
        // Degistirebilecegin parametreler.
        const int N = 512;                  // 2'nin kuvveti
        double genlik = 2.0;
        double fazDerece = 30.0;
        double frekans = 5.0;               // Hz
        double orneklemeFrekansi = 512.0;   // Hz

        // Vektor once olusturulur, fonksiyona gonderilir.
        vector<double> veri(N);

        sinusOlustur(
            veri,
            genlik,
            fazDerece,
            frekans,
            orneklemeFrekansi
        );

        cout << "Vektor uzunlugu: " << N << endl;
        cout << "Genlik: " << genlik << endl;
        cout << "Faz: " << fazDerece << " derece" << endl;
        cout << "Frekans: " << frekans << " Hz" << endl;
        cout << "Periyot: " << 1.0 / frekans << " saniye" << endl;
        cout << "Ornekleme frekansi: "
            << orneklemeFrekansi << " Hz" << endl;

        // Vektor degerlerini konsola ve CSV'ye yaz.
        ofstream dosya("gorev1_sinus_verileri.csv");

        if (!dosya)
            throw runtime_error("CSV dosyasi olusturulamadi.");

        dosya << "ornek,zaman_s,deger\n";

        for (int n = 0; n < N; n++)
        {
            double zaman = n / orneklemeFrekansi;

            cout << "x[" << n << "] = " << veri[n] << endl;

            dosya << n << ","
                << zaman << ","
                << veri[n] << "\n";
        }

        dosya.close();

        Mat vektorGrafigi = grafikCiz(
            veri, orneklemeFrekansi, genlik, true
        );

        Mat sinusGrafigi = grafikCiz(
            veri, orneklemeFrekansi, genlik, false
        );

        imwrite("gorev1_vektor.png", vektorGrafigi);
        imwrite("gorev1_sinus.png", sinusGrafigi);

        imshow("Vektor Ornekleri", vektorGrafigi);
        imshow("Sinus Dalgasi", sinusGrafigi);

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}