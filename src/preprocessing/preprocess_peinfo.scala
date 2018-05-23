import com.datastax.spark.connector._
import play.api.libs.json.Json
import play.api.libs.json._
import java.io.{ByteArrayOutputStream, ByteArrayInputStream}
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

case class PeinfoServiceName(service_name: String, sha256: String)
case class PeinfoSha256(sha256: String, service_name: String, results: Array[Byte])
case class PeinfoJoin(sha256: String, service_name: String, results: String)

def unzip(x: Array[Byte]) : String = {
    val inputStream = new GZIPInputStream(new ByteArrayInputStream(x));
    return scala.io.Source.fromInputStream(inputStream).mkString;
}

val peinfo_service_name_rdd = sc.cassandraTable[PeinfoServiceName](KEYSPACE, SERVICE_NAME_TABLE)
                            .where("service_name=?", "peinfo")
                            .keyBy(x => (x.sha256, x.service_name));
val peinfo_sha256_rdd = sc.cassandraTable[PeinfoSha256](KEYSPACE, SHA256_TABLE)
                      .where("service_name=?", "peinfo")
                      .keyBy(x => (x.sha256, x.service_name));
val peinfo_join = peinfo_service_name_rdd.join(peinfo_sha256_rdd)
                .map(x => (new PeinfoJoin(x._1._1, x._1._2, unzip(x._2._2.results))))
                .distinct();

peinfo_join.toDF().write.format("parquet").save("peinfo.df");

